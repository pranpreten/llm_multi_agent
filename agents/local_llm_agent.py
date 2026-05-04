"""
Local LLM adapter for the MAS framework.
Supports multiple backends: transformers (if model in HF format/ggml/gguf), Ollama, or a simple mock for testing.

This adapter provides a uniform `generate` method returning structured JSON (tool, reason).

Notes:
- This file does not install runtime dependencies. You must have a local runtime available (e.g. Ollama, `transformers` with proper backend, or other).
- The adapter attempts to import optional libraries and will raise clear errors if the chosen backend is not available.
"""

from typing import Optional, Dict, Any
import json
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class LocalLLMAgent:
    def __init__(self, backend: str = 'mock', model_name: Optional[str] = None):
        """
        backend: one of 'mock', 'transformers', 'ollama'
        model_name: local model identifier (path or model id)
        """
        self.backend = backend
        self.model_name = model_name
        logging.info(f"Initializing LocalLLMAgent with backend={backend}, model_name={model_name}")

        if backend == 'transformers':
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize transformers backend: {e}")
        elif backend == 'ollama':
            # Prefer the official Python client if available, otherwise fall back to the
            # Ollama CLI via subprocess (requires `ollama` installed and daemon running).
            try:
                import ollama
                self.ollama = ollama
                self.use_ollama_cli = False
            except Exception:
                logging.warning("Python 'ollama' client not available, will try the 'ollama' CLI via subprocess.")
                self.ollama = None
                self.use_ollama_cli = True
        elif backend == 'mock':
            logging.info("Using mock backend - useful for testing without a model.")
        else:
            raise ValueError("Unsupported backend for LocalLLMAgent. Choose 'mock', 'transformers', or 'ollama'.")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Generate a response for the given prompt. Returns a dictionary with keys: 'raw', 'tool', 'reason'.
        For structured responses, the LLM is expected to return JSON like: {"tool":"tool_name","reason":"..."}
        If the model returns plain text, the adapter will try to parse a JSON object in the text.
        """
        if self.backend == 'mock':
            # Simple sequence-based mock: follow a fixed order
            # Count completed steps by looking for the step number pattern
            import re
            lower = prompt.lower()
            step_match = re.search(r'current step: (\d+)', lower)
            if step_match:
                step_num = int(step_match.group(1))
                
                if step_num == 1:
                    resp = {'tool': 'load_and_inspect_data', 'reason': 'Starting workflow by loading data.', 'finish': False}
                elif step_num == 2:
                    resp = {'tool': 'preprocess_data', 'reason': 'Data loaded, now preprocessing.', 'finish': False}
                elif step_num == 3:
                    resp = {'tool': 'analyze_data', 'reason': 'Data preprocessed, now analyzing.', 'finish': False}
                elif step_num == 4:
                    resp = {'tool': 'generate_recommendations', 'reason': 'Analysis complete, generating recommendations.', 'finish': True}
                else:
                    # Default to next step in sequence
                    resp = {'tool': 'generate_recommendations', 'reason': 'Workflow complete.', 'finish': True}
            else:
                # Fallback: start with loading data
                resp = {'tool': 'load_and_inspect_data', 'reason': 'Starting workflow by loading data.', 'finish': False}
            
            return {'raw': json.dumps(resp), 'tool': resp['tool'], 'reason': resp['reason'], 'parsed': resp}

        if self.backend == 'transformers':
            # Minimal, non-optimized inference flow for local transformers model.
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
            outputs = self.model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Try to parse JSON in output
            parsed = self._extract_json_from_text(text)
            if parsed:
                return {'raw': text, 'tool': parsed.get('tool'), 'reason': parsed.get('reason'), 'parsed': parsed}
            return {'raw': text, 'tool': text.strip(), 'reason': ''}

        if self.backend == 'ollama':
            # Use ollama client to generate text; assumes ollama installed and running
            try:
                if self.ollama is not None and not getattr(self, 'use_ollama_cli', False):
                    # Official Python client path. format='json' forces structured output;
                    # think=False suppresses qwen3-style reasoning blocks.
                    try:
                        response = self.ollama.generate(
                            self.model_name, prompt,
                            format='json',
                            think=False,
                            options={'num_predict': max_tokens, 'temperature': temperature},
                        )
                    except TypeError:
                        # Older ollama client without `think` kwarg
                        response = self.ollama.generate(
                            self.model_name, prompt,
                            format='json',
                            options={'num_predict': max_tokens, 'temperature': temperature},
                        )
                    # Modern ollama client returns dict with 'response' field
                    if isinstance(response, dict):
                        text = (
                            response.get('response')
                            or response.get('output')
                            or response.get('text')
                            or ''
                        )
                    elif hasattr(response, 'response'):
                        text = response.response
                    else:
                        text = str(response)
                else:
                    # Fallback to calling the `ollama` CLI: `ollama run <model> <prompt>`
                    import subprocess
                    proc = subprocess.run(["ollama", "run", self.model_name, prompt], capture_output=True, text=True)
                    if proc.returncode != 0:
                        raise RuntimeError(f"Ollama CLI failed: {proc.stderr.strip()}")
                    text = proc.stdout.strip()
                parsed = self._extract_json_from_text(text)
                if parsed:
                    return {'raw': text, 'tool': parsed.get('tool'), 'reason': parsed.get('reason'), 'parsed': parsed}
                return {'raw': text, 'tool': text.strip(), 'reason': ''}
            except Exception as e:
                raise RuntimeError(f"Ollama generation failed: {e}")

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Find all balanced top-level {...} blocks and try to parse from last to first.
        Reasoning models (e.g., qwen3) often emit thinking + JSON + prose + final JSON;
        the last parseable block is usually the final answer.
        """
        candidates = []
        depth = 0
        start_idx = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx != -1:
                        candidates.append(text[start_idx:i + 1])
                        start_idx = -1
        for cand in reversed(candidates):
            try:
                return json.loads(cand)
            except Exception:
                continue
        return None


if __name__ == '__main__':
    agent = LocalLLMAgent(backend='mock')
    r = agent.generate("Start a workflow to analyze anomalies in the dataset and recommend actions.")
    print(r)
