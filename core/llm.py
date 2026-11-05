from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
from time import perf_counter
import math
import os

import requests
from concurrent.futures import ThreadPoolExecutor

# Concurrent API requests per invocation, modelling a rate-limited endpoint.
# 0 sends every prompt of a batch at once.
MAX_CONCURRENCY = int(os.environ.get('SEM_QUERY_MAX_CONCURRENCY', '0'))


def request_waves(count, calls=1):
	"""Sequential request waves needed to send `count` prompts over `calls` invocations."""
	if count <= 0 or calls <= 0:
		return 0
	if MAX_CONCURRENCY <= 0:
		return calls
	return calls * math.ceil(count / calls / MAX_CONCURRENCY)


class VLLMWrapper:
	"""Thin wrapper for vLLM's OpenAI-compatible completion API."""

	def __init__(self, model_name: str, url: str, api_key: str = "EMPTY", timeout: Optional[float] = None):
		self.model_name = model_name
		self.url = url.rstrip("/")
		self.api_key = api_key
		self.timeout = timeout

	def _parse_usage_tokens(self, response_json: Dict[str, Any]) -> Dict[str, int]:
		usage = response_json.get("usage", {})
		if not isinstance(usage, dict):
			return {"input_tokens": 0, "output_tokens": 0}
		input_tokens = usage.get("prompt_tokens", 0)
		output_tokens = usage.get("completion_tokens", 0)
		return {
			"input_tokens": int(input_tokens) if isinstance(input_tokens, (int, float)) else 0,
			"output_tokens": int(output_tokens) if isinstance(output_tokens, (int, float)) else 0,
		}

	def _check_response(self, response, payload):
		"""Keep the server's rejection reason and request sizes in HTTP errors."""
		try:
			response.raise_for_status()
		except requests.HTTPError as exc:
			message_chars = [
				len(str(message.get("content", "")))
				for message in payload.get("messages", [])
			]
			raise requests.HTTPError(
				f"{exc} | model={payload.get('model')} "
				f"max_tokens={payload.get('max_tokens')} message_chars={message_chars} "
				f"| server_response={response.text}",
				response=response,
				request=exc.request,
			) from exc

	def _error_substitute(self, index: int, exc: Exception) -> str:
		return f"[LLM_ERROR idx={index}] {type(exc).__name__}: {exc}"

	def llm_call(self, prompts: List[Union[str, Sequence[Dict[str, str]]]], **vllm_args: Any) -> Dict[str, Any]:
		if not isinstance(prompts, list):
			raise TypeError("prompts must be a list")

		start_ts = perf_counter()

		if len(prompts) == 0:
			return {
				"responses": [],
				"input_tokens": 0,
				"output_tokens": 0,
				"runtime_sec": perf_counter() - start_ts,
			}

		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}
		responses: List[Optional[str]] = [None] * len(prompts)
		total_input_tokens = 0
		total_output_tokens = 0
		if all(isinstance(p, Sequence) and not isinstance(p, str) for p in prompts):
			for idx, messages in enumerate(prompts):
				try:
					payload = {
						"model": self.model_name,
						"messages": list(messages),
						**vllm_args,
					}
					response = requests.post(
						f"{self.url}",
						json=payload,
						headers=headers,
						timeout=self.timeout,
					)
					self._check_response(response, payload)

					data = response.json()
					tokens = self._parse_usage_tokens(data)
					total_input_tokens += tokens["input_tokens"]
					total_output_tokens += tokens["output_tokens"] # Debug: print the full response JSON

					choices = data.get("choices", [])
					if not isinstance(choices, list) or len(choices) == 0:
						raise ValueError(f"Invalid vLLM chat response: missing choices in {data}")

					first = choices[0]
					if not isinstance(first, dict):
						raise ValueError(f"Invalid vLLM chat response choice: {first}")

					message = first.get("message", {})
					if isinstance(message, dict):
						responses[idx] = message.get("content", "") or ""
					else:
						responses[idx] = ""
				except Exception as exc:
					print(exc)
					responses[idx] = self._error_substitute(idx, exc)

			for i in range(len(responses)):
				if responses[i] is None:
					responses[i] = f"[LLM_ERROR idx={i}] Missing response for prompt index {i}"

			return {
				"responses": [item if item is not None else "" for item in responses],
				"input_tokens": total_input_tokens,
				"output_tokens": total_output_tokens,
				"runtime_sec": perf_counter() - start_ts,
			}

		raise TypeError("prompts must be a list of chat message lists")

	def _single_request(self, payload, headers):
		try:
			response = requests.post(
				f"{self.url}",
				json=payload,
				headers=headers,
				timeout=self.timeout,
			)
			self._check_response(response, payload)
			return response.json()
		except Exception as exc:
			return exc

	def _as_llm_call(self, prompts: List[List[Dict[str, str]]], **vllm_args: Any) -> Dict[str, Any]:
		print("Using : deepseek")
		start_ts = perf_counter()
		if len(prompts) == 0:
			return {
				"responses": [],
				"input_tokens": 0,
				"output_tokens": 0,
				"runtime_sec": perf_counter() - start_ts,
			}
		headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

		# 1. Prepare all payloads
		payloads = [
			{"model": self.model_name, "messages": list(p), **vllm_args}
			for p in prompts
		]

		if "deepseek" in self.url:
			# For DeepSeek OpenAI completions API
			for payload in payloads:
				payload["extra_body"] = {"thinking": {"type": "disabled"}}
				payload["reasoning_effort"] = "none"
				if "max_tokens" in payload:
					payload.pop("max_tokens")
				if "temperature" in payload:
					payload.pop("temperature")
		elif "Qwen" in self.model_name and not "Instruct" in self.model_name:
			# disable reasoning to control token consumption
			for payload in payloads:
				payload["chat_template_kwargs"] =  {"enable_thinking": False}

		# 2. Use a ThreadPool to send requests in parallel
		# Max_workers can be tuned, but 10-20 is usually plenty for vLLM
		responses_data = []
		with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
			# Map the helper function across all payloads
			results = list(executor.map(lambda p: self._single_request(p, headers), payloads))

		# 3. Process results (same logic as before)
		final_responses = []
		total_in = 0
		total_out = 0

		for idx, data in enumerate(results):
			if isinstance(data, Exception):
				print(f"Error in prompt {idx}: {data}")
				final_responses.append(self._error_substitute(idx, data))
			else:
				tokens = self._parse_usage_tokens(data)
				total_in += tokens["input_tokens"]
				total_out += tokens["output_tokens"]
				content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
				final_responses.append(content)
		print(final_responses)
		return {
			"responses": final_responses,
			"input_tokens": total_in,
			"output_tokens": total_out,
			"runtime_sec": perf_counter() - start_ts,
		}


	def _as_llm_call_azure(
		self,
		prompts: List[List[Dict[str, str]]],
		**vllm_args: Any,
	) -> Dict[str, Any]:
		start_ts = perf_counter()

		if len(prompts) == 0:
			return {
				"responses": [],
				"input_tokens": 0,
				"output_tokens": 0,
				"runtime_sec": perf_counter() - start_ts,
			}

		headers = {
			"Content-Type": "application/json",
		}
		headers["api-key"] = self.api_key

		args = dict(vllm_args)

		if "max_tokens" in args:
			args.pop("max_tokens")

		if "temperature" in args:
			args.pop("temperature")

		# 1. Prepare payloads
		payloads = [
			{
				**args,
				"model": self.model_name,
				"input": list(p),
				"reasoning": {
					"effort": "none"
				},
				"text": {
					"verbosity": "low"
				}
			}
			for p in prompts
		]

		# 2. Send requests concurrently
		workers = min(len(prompts), MAX_CONCURRENCY) if MAX_CONCURRENCY > 0 else len(prompts)
		with ThreadPoolExecutor(max_workers=workers) as executor:
			results = list(
				executor.map(
					lambda p: self._single_request(p, headers),
					payloads,
				)
			)

		# 3. Process results
		final_responses = []
		total_in = 0
		total_out = 0

		def extract_azure_text(data):
			return "".join(
				part.get("text", "")
				for item in data.get("output", [])
				if item.get("type") == "message"
				for part in item.get("content", [])
				if part.get("type") == "output_text"
			)

		for idx, data in enumerate(results):
			if isinstance(data, Exception):
				print(f"Error in prompt {idx}: {data}")
				final_responses.append(
					self._error_substitute(idx, data)
				)
				continue

			try:
				usage = data.get("usage") or {}
				total_in += usage.get("input_tokens", 0)
				total_out += usage.get("output_tokens", 0)
				# print(data)

				content = extract_azure_text(data)
				# print(f"Prompt {idx} response: {content}")

				if data.get("status") == "incomplete":
					print(
							f"Warning: prompt {idx} incomplete: "
							f"{data.get('incomplete_details')}"
					)

				else:
					tokens = self._parse_usage_tokens(data)
					total_in += tokens["input_tokens"]
					total_out += tokens["output_tokens"]

				final_responses.append(content or "")

			except Exception as exc:
				print(f"Error parsing prompt {idx}: {exc}")
				final_responses.append(
					self._error_substitute(idx, exc)
				)

		print(final_responses)
		return {
			"responses": final_responses,
			"input_tokens": total_in,
			"output_tokens": total_out,
			"runtime_sec": perf_counter() - start_ts,
		}

	def as_llm_call(self, prompts: List[List[Dict[str, str]]], **vllm_args: Any) -> Dict[str, Any]:
		if self.url.startswith("https://") and "azure" in self.url:
			return self._as_llm_call_azure(prompts, **vllm_args)
		else:
			return self._as_llm_call(prompts, **vllm_args)