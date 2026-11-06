"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
import time
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""
        self._consecutive_tool_errors = 0

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        system_information = {
            "cwd": self.env.cwd,
            "machine": self.env.machine,
            "release": self.env.release,
            "system": self.env.system,
            "version": self.env.version,
        }
        self.system_prompt: str = (
            "You are a coding agent tasked with fixing software issues.\n"
            "Use the execute tool to inspect the repository, reproduce the issue, "
            "make appropriate changes, and run relevant tests.\n"
            "Inspect tool results before deciding your next action. "
            "If an action fails, use the observation to adjust your approach.\n"
            "call send_message with your final summary to end the run.\n"
            "Report verified results accurately and distinguish them from assumptions.\n\n"
            "<system_information>\n"
            f"{json.dumps(system_information, indent=2)}\n"
            "</system_information>"
        )
        self.task_prompt = task

        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            catalog = "\n".join(
                f"- {name}: {skill['metadata']}"
                for name, skill in sorted(self.skills.items())
            )
            self.system_prompt += (
                "\n\nAvailable skills (load with invoke_skill before using):\n" + catalog
            )
    def query_language_model(self):
        if self._consecutive_tool_errors >= 6:
            raise RuntimeError("Aborting after 6 consecutive invalid tool calls")
        return super().query_language_model()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        
        # Defer the abort until the next action so the last linked observations
        # are retained by the shared loop and saved in the trajectory.
        if self._consecutive_tool_errors >= 6:
            raise RuntimeError("Aborting after 6 consecutive invalid tool calls")
        messages = []
        for call in tool_calls:
            failed = False
            try:
                if self.finished:
                    raise ValueError("The task has already been submitted")
                if self._consecutive_tool_errors >= 6:
                    raise ValueError("Failure limit reached; no further tools will execute")
                function = call["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")

                if name == "execute":
                    schema = EXECUTE_TOOL["function"]["parameters"]
                    unexpected = set(arguments) - set(schema["properties"])
                    if unexpected:
                        raise ValueError(
                            "Invalid execute arguments: unexpected parameter names "
                            f"{sorted(unexpected)!r}. "
                            'Put the entire shell script in the "command" value. '
                            'Example: {"command": "pwd; ls -la"}.'
                        )
                    command = arguments.get("command")
                    if not isinstance(command, (str, list)) or not command:
                        raise ValueError("command must be a non-empty string or argv list")
                    if isinstance(command, list) and not all(isinstance(arg, str) for arg in command):
                        raise ValueError("Every command argv entry must be a string")
                    print(
                        f"[execute] timeout={arguments.get('timeout')} "
                        f"cwd={arguments.get('cwd') or self.env.cwd}: "
                        f"{arguments.get('command')}",
                        flush=True,
                    )
                    started = time.monotonic()
                    result = self.env.execute(**arguments)
                    print(
                        f"[execute] finished in {time.monotonic() - started:.1f}s; "
                        f"returncode={result.get('returncode')}; "
                        f"exception={result.get('exception_info') or 'none'}",
                        flush=True,
                    )
                    content = format_tool_output(result)
                elif name == "send_message":
                    summary = arguments["summary"]
                    if not isinstance(summary, str):
                        raise ValueError("summary must be a string")
                    patch_result = self.env.execute("cat patch.txt")
                    patch = patch_result.get("stdout", patch_result.get("output", ""))
                    if patch_result.get("returncode") != 0 or not patch.strip():
                        raise ValueError(
                            "Submission rejected: patch.txt is missing, unreadable, or empty. "
                            "Create and verify the required patch before calling send_message again."
                        )
                    self.submitted_patch = patch
                    content = summary
                    self.finished = True
                elif name == "invoke_skill":
                    content = self.skills[arguments["name"]]["content"]
                else:
                    raise ValueError(f"Unknown tool {name!r}")
            except (KeyError, TypeError, ValueError) as exc:
                content = f"Error: {exc}"
                failed = True

            if failed:
                self._consecutive_tool_errors += 1
                if self._consecutive_tool_errors >= 3:
                    content += (
                        '\nRecovery: send one simple execute call: {"command": "pwd"}. '
                        'Put the entire script in command; do not add a bash -lc wrapper. '
                        f'Consecutive invalid calls: {self._consecutive_tool_errors}/6.'
                    )
            else:
                self._consecutive_tool_errors = 0

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": content,
            })
        return messages
