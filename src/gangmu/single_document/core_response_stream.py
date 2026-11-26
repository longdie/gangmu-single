"""Bounded Responses SSE transport; no supplier calls from this parser.

Only a fully framed semantic response.completed carries a successful Response.
Text deltas, item.done, [DONE], EOF and parseable accumulated output never do.
The host still applies its authoritative _output_object to the returned Response.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
from typing import Any, Callable

from .semantic_inspection import redact_semantic_value_bounded


VERSION = 'gangmu.responses.semantic_sse.draft.v1'


class StreamProtocolError(ValueError):
    """Nonretryable protocol or configured resource-limit failure."""


class StreamTruncatedError(ConnectionError):
    """No fully framed semantic completion; follows existing bounded retry gate."""


class StreamTransientError(ConnectionError):
    """An explicitly retryable provider failure, never a successful Response."""

    def __init__(self, error: dict):
        self.error_code = error['code']
        self.error_type = error['type']
        self.provider_message = error.get('message')
        # Only an allowlisted code enters the exception text. Provider details
        # remain available to the caller and in the redacted raw transcript.
        super().__init__('stream_provider_transient_error_' + self.error_code)


@dataclass(frozen=True)
class StreamLimits:
    max_response_bytes: int = 32 * 1024 * 1024
    max_line_bytes: int = 16 * 1024 * 1024
    max_event_bytes: int = 16 * 1024 * 1024
    max_events: int = 100_000
    read_chunk_bytes: int = 16_384

    def __post_init__(self):
        for value in vars(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError('stream limits must be positive integers')
        if self.max_line_bytes > self.max_response_bytes or self.max_event_bytes > self.max_response_bytes:
            raise ValueError('event and line bounds must not exceed the response bound')


def _json(value):
    def invalid_constant(_):
        raise StreamProtocolError('stream_json_nonfinite_number')
    try:
        return json.loads(value, parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise StreamProtocolError('stream_json_malformed') from exc


def redact_semantic_value(value: Any, redact: Callable[[Any], Any], *, depth=0) -> tuple[Any, bool]:
    """Inspect all semantic content under separate structure/encoding limits.

    Historical explicit depth reserves both budgets conservatively.  Any
    redaction or uninspectable resource failure remains a failed response.
    """
    return redact_semantic_value_bounded(value, redact, initial_depth=depth)


def _safe_framed_sse_raw(raw: bytes, redact, *, eof_seen: bool) -> tuple[bytes, dict]:
    """Only complete safely inspected frames are persisted; unsafe suffix omitted."""
    boundaries, line_start = [], 0
    for match in re.finditer(rb'\r\n|\r|\n', raw):
        if match.group() == b'\r' and match.end() == len(raw) and not eof_seen:
            break
        if match.start() == line_start:
            boundaries.append(match.end())
        line_start = match.end()
    output = bytearray()
    start, retained_original, changed = 0, 0, False
    omission_reason = None
    for end in boundaries:
        chunk = raw[start:end]
        try:
            text = chunk.decode('utf-8')
            lines = re.split(r'\r\n|\r|\n', text)
            data_lines, nondata = [], []
            field_changed = False
            for line in lines:
                if not line:
                    continue
                name, separator, item = line.partition(':')
                if separator and item.startswith(' '):
                    item = item[1:]
                if name == 'data':
                    data_lines.append(item)
                else:
                    safe_line, line_changed = redact_semantic_value(line, redact)
                    nondata.append(safe_line)
                    field_changed |= line_changed
            if data_lines:
                data = '\n'.join(data_lines)
                if data == '[DONE]':
                    safe_data, data_changed = data, False
                else:
                    decoded = _json(data)
                    safe_decoded, data_changed = redact_semantic_value(decoded, redact)
                    safe_data = json.dumps(safe_decoded, ensure_ascii=False, allow_nan=False)
                if data_changed or field_changed:
                    chunk = ('\n'.join(nondata + ['data: ' + safe_data]) + '\n\n').encode()
                    changed = True
            elif field_changed:
                chunk = ('\n'.join(nondata) + '\n\n').encode()
                changed = True
        except (ValueError, UnicodeError, RecursionError):
            omission_reason = 'unsafe_malformed_sse_frame_and_following_suffix'
            break
        output.extend(chunk)
        start = end
        retained_original = end
    if retained_original < len(raw) and omission_reason is None:
        omission_reason = 'unsafe_unfinished_sse_frame_suffix'
    omitted = len(raw) - retained_original
    return bytes(output), {'omitted_original_bytes': omitted, 'omission_reason': omission_reason,
                           'semantic_secret_redaction_changed': changed,
                           'retained_original_prefix_bytes': retained_original}


@dataclass
class ResponseCapture:
    """Entire *observed* prefix, bounded; may end at the semantic terminal event.

    original_bytes are never persisted directly. Public retention must pass the
    resolved-credential redactor over the complete observed byte string before
    encoding it as base64, including secrets split across chunks or SSE frames.
    """
    mode: str
    limits: StreamLimits = field(default_factory=StreamLimits)
    original_bytes: bytearray = field(default_factory=bytearray)
    observed_digest: Any = field(default_factory=hashlib.sha256, repr=False)
    bytes_seen: int = 0
    captured_bytes_complete: bool = True
    events_seen: int = 0
    event_types: dict[str, int] = field(default_factory=dict)
    terminal_event: str | None = None
    response: dict | None = None
    response_id: str | None = None
    last_sequence: int | None = None
    stopped_after_semantic_terminal: bool = False
    post_terminal_partial_frame_ignored: bool = False
    eof_seen: bool = False
    failure: str | None = None
    raw_format: str = 'sse'

    def append(self, data: bytes):
        if not isinstance(data, bytes):
            raise StreamProtocolError('stream_reader_did_not_return_bytes')
        self.bytes_seen += len(data)
        self.observed_digest.update(data)
        remaining = self.limits.max_response_bytes - len(self.original_bytes)
        self.original_bytes.extend(data[:max(remaining, 0)])
        if len(data) > remaining:
            self.captured_bytes_complete = False
            raise StreamProtocolError('stream_response_byte_limit')

    def public(self, redact: Callable[[Any], Any]) -> dict:
        raw = bytes(self.original_bytes)
        if self.raw_format == 'sse':
            public_raw, safety = _safe_framed_sse_raw(raw, redact, eof_seen=self.eof_seen)
        elif self.eof_seen and self.captured_bytes_complete:
            try:
                text = raw.decode('utf-8')
                try:
                    decoded = _json(text)
                except ValueError:
                    decoded = text
                safe, changed = redact_semantic_value(decoded, redact)
                public_raw = (json.dumps(safe, ensure_ascii=False, allow_nan=False).encode()
                              if changed and not isinstance(decoded, str) else safe.encode() if changed else raw)
                safety = {'omitted_original_bytes': 0, 'omission_reason': None,
                          'semantic_secret_redaction_changed': changed,
                          'retained_original_prefix_bytes': len(raw)}
            except (ValueError, UnicodeError, RecursionError):
                public_raw = b''
                safety = {'omitted_original_bytes': len(raw), 'omission_reason': 'unsafe_http_body_encoding',
                          'semantic_secret_redaction_changed': False, 'retained_original_prefix_bytes': 0}
        else:
            public_raw = b''
            safety = {'omitted_original_bytes': len(raw), 'omission_reason': 'unsafe_unfinished_or_capped_http_body',
                      'semantic_secret_redaction_changed': False, 'retained_original_prefix_bytes': 0}
        metadata = {
            'schema_version': VERSION,
            'mode': self.mode,
            'raw_encoding': 'base64_after_resolved_secret_redaction',
            'raw_base64': base64.b64encode(public_raw).decode('ascii'),
            'redacted_raw_sha256': hashlib.sha256(public_raw).hexdigest(),
            'observed_raw_sha256': self.observed_digest.hexdigest(),
            'captured_original_prefix_sha256': hashlib.sha256(raw).hexdigest(),
            'raw_redaction_changed': public_raw != raw,
            'raw_format': self.raw_format,
            'raw_safety': safety,
            'bytes_seen': self.bytes_seen,
            'captured_bytes': len(raw),
            'captured_bytes_complete': self.captured_bytes_complete,
            'limits': vars(self.limits),
            'events_seen': self.events_seen,
            'event_types': self.event_types,
            'terminal_event': self.terminal_event,
            'response_id': self.response_id,
            'last_sequence': self.last_sequence,
            'stopped_after_semantic_terminal': self.stopped_after_semantic_terminal,
            'post_terminal_partial_frame_ignored': self.post_terminal_partial_frame_ignored,
            'eof_seen': self.eof_seen,
            'failure': self.failure,
        }
        # Raw base64 has already been redacted; do not feed the encoded data
        # through a string sanitizer again and corrupt its reversible form.
        safe_metadata, metadata_changed = redact_semantic_value({k: v for k, v in metadata.items() if k != 'raw_base64'}, redact)
        if not isinstance(safe_metadata, dict) or not isinstance(safe_metadata.get('raw_safety'), dict):
            # Inspection can omit the root or redact a control key.  Preserve
            # a fixed safe shape so finally records failure and actual usage.
            safe_metadata = {'schema_version': VERSION,
                'failure': 'response_metadata_secret_inspection_failed',
                'raw_safety': {'semantic_secret_redaction_changed': True,
                               'omission_reason': 'uninspectable_response_metadata'}}
        elif metadata_changed:
            safe_metadata['raw_safety']['semantic_secret_redaction_changed'] = True
        return {**redact(safe_metadata),
                'raw_base64': metadata['raw_base64']}


class SemanticSSEParser:
    def __init__(self, capture: ResponseCapture):
        if capture.mode != 'sse':
            raise ValueError('semantic SSE parser requires the SSE capture mode')
        self.capture = capture
        self.pending = bytearray()
        self.data_lines: list[str] = []
        self.event_name: str | None = None
        self.event_bytes = 0
        self.first_line = True
        self.sentinel_seen = False
        self.output_text: dict[tuple, list[str]] = {}
        self.output_candidates: list[str] = []
        self.output_done = False

    @staticmethod
    def _message_output(item: Any) -> tuple[list[str], bool]:
        if (not isinstance(item, dict) or item.get('type') != 'message'
                or not isinstance(item.get('content'), list)):
            return [], False
        texts, refusal = [], False
        for part in item['content']:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'output_text' and isinstance(part.get('text'), str):
                texts.append(part['text'])
            refusal |= isinstance(part.get('refusal'), str) and bool(part['refusal'].strip())
        return [*texts, ''.join(texts)], refusal

    @staticmethod
    def _output_key(event: dict, content_index: Any, item_id: Any = None) -> tuple:
        index = event.get('output_index')
        identity = ('index', index) if type(index) is int else (
            'item', item_id if isinstance(item_id, str) else event.get('item_id'))
        if not isinstance(identity[1], (str, int)):
            identity = ('item', None)
        return (*identity, content_index if type(content_index) is int else None)

    def _seed_message_output(self, item: Any, event: dict):
        if (not isinstance(item, dict) or item.get('type') != 'message'
                or not isinstance(item.get('content'), list)):
            return
        for index, part in enumerate(item['content']):
            if (isinstance(part, dict) and part.get('type') == 'output_text'
                    and isinstance(part.get('text'), str) and part['text']):
                key = self._output_key(event, index, item.get('id'))
                self.output_text.setdefault(key, [part['text']])

    def _observe_output(self, event: dict):
        event_type = event['type']
        snapshot = event.get('response')
        if isinstance(snapshot, dict) and isinstance(snapshot.get('output'), list):
            for index, item in enumerate(snapshot['output']):
                texts, refusal = self._message_output(item)
                self.output_candidates.extend(texts)
                self.output_done |= refusal or (isinstance(item, dict) and item.get('status') == 'completed'
                                                and any(text.strip() for text in texts))
                self._seed_message_output(item, {'output_index': index})
        if event_type == 'response.output_text.delta' and isinstance(event.get('delta'), str):
            key = self._output_key(event, event.get('content_index'))
            self.output_text.setdefault(key, []).append(event['delta'])
        if event_type == 'response.output_text.done' and isinstance(event.get('text'), str):
            self.output_done |= bool(event['text'].strip())
        if event_type in ('response.output_item.added', 'response.output_item.done'):
            texts, refusal = self._message_output(event.get('item'))
            self.output_candidates.extend(texts)
            self.output_done |= refusal or (event_type == 'response.output_item.done'
                                            and any(text.strip() for text in texts))
            if event_type.endswith('.added'):
                self._seed_message_output(event.get('item'), event)
        if event_type in ('response.content_part.added', 'response.content_part.done'):
            part = event.get('part')
            if isinstance(part, dict):
                if part.get('type') == 'output_text' and isinstance(part.get('text'), str):
                    self.output_candidates.append(part['text'])
                    self.output_done |= event_type.endswith('.done') and bool(part['text'].strip())
                    if event_type.endswith('.added') and part['text']:
                        key = self._output_key(event, event.get('content_index'))
                        self.output_text.setdefault(key, [part['text']])
                self.output_done |= isinstance(part.get('refusal'), str) and bool(part['refusal'].strip())
        if event_type in ('response.refusal.delta', 'response.refusal.done'):
            text = event.get('delta') if event_type.endswith('.delta') else event.get('refusal')
            self.output_done |= isinstance(text, str) and bool(text.strip())

    def _has_answer_output(self, snapshot: dict | None) -> bool:
        """Conservatively prevent repurchasing output; never authorize success.

        A complete JSON prefix remains substantive even if later bytes make
        the combined text invalid. Separate content streams cannot hide it.
        The actual host schema and response.completed are still authoritative.
        """
        if self.output_done:
            return True
        candidates = list(self.output_candidates)
        messages: dict[tuple, list[tuple[int, str]]] = {}
        for key, parts in self.output_text.items():
            text = ''.join(parts)
            candidates.append(text)
            messages.setdefault(key[:2], []).append((key[2] if key[2] is not None else -1, text))
        candidates.extend(''.join(text for _, text in sorted(parts)) for parts in messages.values())
        if isinstance(snapshot, dict) and isinstance(snapshot.get('output'), list):
            for item in snapshot['output']:
                texts, refusal = self._message_output(item)
                if refusal or (isinstance(item, dict) and item.get('status') == 'completed'
                               and any(text.strip() for text in texts)):
                    return True
                candidates.extend(texts)
        def invalid_constant(_):
            raise ValueError('nonfinite output')
        decoder = json.JSONDecoder(parse_constant=invalid_constant)
        for text in candidates:
            try:
                decoder.raw_decode(text.lstrip())
            except (ValueError, TypeError, RecursionError):
                continue
            return True
        return False

    def _transient_error(self, event_type: str, error: Any, snapshot: dict | None):
        allowed = {
            ('error', 'server_is_overloaded', 'service_unavailable_error'),
            ('response.failed', 'upstream_stream_break', 'upstream_error'),
        }
        if (isinstance(error, dict)
                and isinstance(error.get('code'), str) and isinstance(error.get('type'), str)
                and (event_type, error.get('code'), error.get('type')) in allowed
                and not self._has_answer_output(snapshot)):
            raise StreamTransientError(error)

    def feed(self, data: bytes):
        self.capture.append(data)
        self.pending.extend(data)
        while True:
            match = re.search(rb'[\r\n]', self.pending)
            if match is None:
                if len(self.pending) > self.capture.limits.max_line_bytes:
                    raise StreamProtocolError('stream_line_byte_limit')
                break
            index = match.start()
            if self.pending[index] == 13 and index + 1 == len(self.pending):
                # A CR/LF pair may straddle HTTP chunks.
                if index > self.capture.limits.max_line_bytes:
                    raise StreamProtocolError('stream_line_byte_limit')
                break
            width = 2 if self.pending[index:index + 2] == b'\r\n' else 1
            line = bytes(self.pending[:index])
            del self.pending[:index + width]
            self._line(line)

    def _line(self, raw: bytes):
        if len(raw) > self.capture.limits.max_line_bytes:
            raise StreamProtocolError('stream_line_byte_limit')
        try:
            line = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise StreamProtocolError('stream_invalid_utf8') from exc
        if self.first_line:
            line = line.removeprefix('\ufeff')
            self.first_line = False
        if not line:
            self._dispatch()
            self.data_lines = []
            self.event_name = None
            self.event_bytes = 0
            return
        if line.startswith(':'):
            return
        self.event_bytes += len(raw)
        if self.event_bytes > self.capture.limits.max_event_bytes:
            raise StreamProtocolError('stream_event_byte_limit')
        name, separator, value = line.partition(':')
        if separator and value.startswith(' '):
            value = value[1:]
        if name == 'data':
            self.data_lines.append(value)
        elif name == 'event':
            if self.event_name is not None and self.event_name != value:
                raise StreamProtocolError('stream_conflicting_event_fields')
            self.event_name = value
        # SSE id/retry/extension fields do not establish semantic completion.

    def _dispatch(self):
        if not self.data_lines:
            if self.event_name:
                raise StreamProtocolError('stream_event_has_no_data')
            return
        data = '\n'.join(self.data_lines)
        if data == '[DONE]':
            if self.capture.terminal_event != 'response.completed':
                raise StreamTruncatedError('stream_sentinel_without_semantic_completion')
            if self.sentinel_seen:
                raise StreamProtocolError('stream_duplicate_done_sentinel')
            self.sentinel_seen = True
            return
        event = _json(data)
        if not isinstance(event, dict) or not isinstance(event.get('type'), str) or not event['type']:
            raise StreamProtocolError('stream_event_missing_type')
        event_type = event['type']
        if self.event_name not in (None, '', event_type):
            raise StreamProtocolError('stream_event_type_conflict')
        if self.capture.terminal_event is not None:
            raise StreamProtocolError('stream_semantic_event_after_terminal')
        self.capture.events_seen += 1
        if self.capture.events_seen > self.capture.limits.max_events:
            raise StreamProtocolError('stream_event_count_limit')
        self.capture.event_types[event_type] = self.capture.event_types.get(event_type, 0) + 1
        if 'sequence_number' in event:
            sequence = event['sequence_number']
            if (type(sequence) is not int or sequence < 0 or
                    (self.capture.last_sequence is not None and sequence <= self.capture.last_sequence)):
                raise StreamProtocolError('stream_event_sequence_invalid')
            self.capture.last_sequence = sequence
        snapshot = event.get('response')
        failure_event = event_type in ('response.failed', 'response.incomplete')
        if failure_event:
            # Provider failure envelopes may omit an id already established by
            # created/in_progress. Keep the original failure and usage without
            # synthesizing an id or relaxing successful Response validation.
            self.capture.terminal_event = event_type
        if isinstance(snapshot, dict):
            identifier = snapshot.get('id')
            if failure_event and 'id' not in snapshot:
                identifier = self.capture.response_id
            elif not isinstance(identifier, str) or not identifier:
                raise StreamProtocolError('stream_response_missing_id')
            if self.capture.response_id is not None and identifier != self.capture.response_id:
                raise StreamProtocolError('stream_response_id_changed')
            self.capture.response_id = identifier
        if failure_event:
            # A conflicting/invalid explicit id must not supply another
            # Response's usage to the caller. Its raw frame is still retained.
            self.capture.response = snapshot if isinstance(snapshot, dict) else None
        self._observe_output(event)
        if failure_event:
            if event_type == 'response.failed' and isinstance(snapshot, dict) and snapshot.get('status') == 'failed':
                self._transient_error(event_type, snapshot.get('error'), snapshot)
            raise StreamProtocolError('stream_response_not_completed_' + event_type)
        if event_type == 'error':
            self.capture.terminal_event = event_type
            self._transient_error(event_type, event.get('error'), snapshot)
            raise StreamProtocolError('stream_provider_error_event')
        if event_type == 'response.completed':
            self.capture.terminal_event = event_type
            self.capture.response = snapshot if isinstance(snapshot, dict) else None
            if (not isinstance(snapshot, dict) or snapshot.get('object') != 'response'
                    or snapshot.get('status') != 'completed'
                    or snapshot.get('error') is not None
                    or snapshot.get('incomplete_details') is not None
                    or not isinstance(snapshot.get('output'), list)):
                raise StreamProtocolError('stream_completed_response_malformed')

    def finish_eof(self):
        self.capture.eof_seen = True
        # A trailing lone CR can terminate a line at EOF, but no unfinished
        # data frame is silently dispatched without its empty-line delimiter.
        if self.pending.endswith(b'\r'):
            self._line(bytes(self.pending[:-1]))
            self.pending.clear()
        if self.capture.terminal_event != 'response.completed':
            raise StreamTruncatedError('stream_eof_without_semantic_completion')
        self._check_terminal_tail()

    def _check_terminal_tail(self):
        # Already framed extra semantic events are rejected in _dispatch.
        # An incomplete *post-terminal* suffix is retained but never contributes
        # to acceptance. This keeps optional [DONE]/keepalive chunk boundaries
        # from invalidating an already fully framed complete Response. No later
        # read is made merely to inspect a post-terminal suffix.
        self.capture.post_terminal_partial_frame_ignored = bool(
            self.pending or self.data_lines or self.event_name)

    def completed_response(self) -> dict | None:
        if self.capture.terminal_event == 'response.completed':
            self._check_terminal_tail()
            self.capture.stopped_after_semantic_terminal = True
            return self.capture.response
        return None


def _set_timeout(stream, remaining: float):
    setter = getattr(stream, 'set_read_timeout', None)
    if callable(setter):
        setter(remaining)
        return
    # HTTPError wraps the ordinary HTTPResponse in another .fp layer. Walk a
    # bounded number of wrappers so error bodies retain the same real socket
    # deadline as successful streams, rather than losing provider diagnostics.
    current = stream
    for _ in range(4):
        for socket_object in (getattr(current, '_sock', None),
                              getattr(getattr(current, 'raw', None), '_sock', None)):
            setter = getattr(socket_object, 'settimeout', None)
            if callable(setter):
                setter(remaining)
                return
        current = getattr(current, 'fp', None)
        if current is None:
            break
    # SSE cannot claim a cumulative deadline for a transport that provides no
    # bounded read. Offline fixtures implement set_read_timeout explicitly.
    raise StreamProtocolError('stream_deadline_control_unavailable')


def read_sse_response(stream, *, capture: ResponseCapture, deadline: float,
                      clock: Callable[[], float] = time.monotonic) -> dict:
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('stream deadline must be finite')
    parser = SemanticSSEParser(capture)
    reader = getattr(stream, 'read1', None)
    if not callable(reader):
        reader = stream.read
    try:
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError('stream_cumulative_deadline')
            _set_timeout(stream, remaining)
            chunk = reader(capture.limits.read_chunk_bytes)
            if deadline - clock() < 0:
                # Retain a delivered late chunk, but never accept its Response.
                if chunk:
                    capture.append(chunk)
                raise TimeoutError('stream_cumulative_deadline')
            if not chunk:
                parser.finish_eof()
                response = parser.completed_response()
                if response is None:
                    raise StreamTruncatedError('stream_no_complete_response')
                return response
            parser.feed(chunk)
            response = parser.completed_response()
            if response is not None:
                return response
    except BaseException as exc:
        capture.failure = str(exc) if isinstance(exc, (StreamProtocolError, StreamTruncatedError, StreamTransientError, TimeoutError)) else (
            'stream_reader_exception_' + type(exc).__name__)
        raise


def read_bounded_body(stream, *, capture: ResponseCapture, deadline: float,
                      clock: Callable[[], float] = time.monotonic) -> bytes:
    """Retain a non-SSE HTTP/error body without treating it as completion."""
    capture.raw_format = 'http_body'
    reader = getattr(stream, 'read1', None)
    if not callable(reader):
        reader = stream.read
    try:
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError('stream_cumulative_deadline')
            _set_timeout(stream, remaining)
            chunk = reader(capture.limits.read_chunk_bytes)
            if chunk:
                capture.append(chunk)
            if deadline - clock() < 0:
                raise TimeoutError('stream_cumulative_deadline')
            if not chunk:
                capture.eof_seen = True
                return bytes(capture.original_bytes)
    except BaseException as exc:
        capture.failure = str(exc) if isinstance(exc, (StreamProtocolError, StreamTruncatedError, StreamTransientError, TimeoutError)) else (
            'stream_reader_exception_' + type(exc).__name__)
        raise


def validate_sse_transcript(transcript: dict, retained_response: dict,
                            redact: Callable[[Any], Any]) -> None:
    """A successful cache/recovery must retain a real semantic completion.

    The public raw transcript has resolved secrets removed; verify its hash and
    replay its complete events without a network transport or ledger mutation.
    Its redacted final Response must match the independently retained Response.
    """
    if (not isinstance(transcript, dict) or transcript.get('schema_version') != VERSION
            or transcript.get('mode') != 'sse'
            or transcript.get('raw_encoding') != 'base64_after_resolved_secret_redaction'
            or transcript.get('terminal_event') != 'response.completed'
            or transcript.get('stopped_after_semantic_terminal') is not True
            or transcript.get('captured_bytes_complete') is not True
            or transcript.get('limits') != vars(StreamLimits())):
        raise StreamProtocolError('retained_sse_transcript_has_no_verified_terminal')
    safety = transcript.get('raw_safety')
    if not isinstance(safety, dict) or safety.get('semantic_secret_redaction_changed') is not False:
        raise StreamProtocolError('retained_sse_semantic_secret_change_not_original_data')
    try:
        raw = base64.b64decode(transcript['raw_base64'], validate=True)
    except (ValueError, TypeError, KeyError) as exc:
        raise StreamProtocolError('retained_sse_raw_encoding_invalid') from exc
    if hashlib.sha256(raw).hexdigest() != transcript.get('redacted_raw_sha256'):
        raise StreamProtocolError('retained_sse_raw_digest_mismatch')
    replay = ResponseCapture('sse')
    parser = SemanticSSEParser(replay)
    parser.feed(raw)
    response = parser.completed_response()
    if response is None or redact(response) != retained_response:
        raise StreamProtocolError('retained_sse_terminal_response_mismatch')
