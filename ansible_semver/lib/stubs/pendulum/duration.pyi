from .constants import SECONDS_PER_DAY as SECONDS_PER_DAY, SECONDS_PER_HOUR as SECONDS_PER_HOUR, SECONDS_PER_MINUTE as SECONDS_PER_MINUTE, US_PER_SECOND as US_PER_SECOND
from datetime import timedelta
from pendulum.utils._compat import PYPY as PYPY, decode as decode
from typing import Any, Optional, Union, overload, Tuple, TypeVar

_Self = TypeVar('_Self', bound=Duration)

class Duration(timedelta):
    def __new__(cls, days: int = ..., seconds: int = ..., microseconds: int = ..., milliseconds: int = ..., minutes: int = ..., hours: int = ..., weeks: int = ..., years: int = ..., months: int = ...) -> Duration: ...
    def total_minutes(self) -> float: ...
    def total_hours(self) -> float: ...
    def total_days(self) -> float: ...
    def total_weeks(self) -> float: ...
    def total_seconds(self) -> float: ...
    @property
    def years(self) -> int: ...
    @property
    def months(self) -> int: ...
    @property
    def weeks(self) -> int: ...
    @property
    def days(self) -> int: ...
    @property
    def remaining_days(self) -> int: ...
    @property
    def hours(self) -> int: ...
    @property
    def minutes(self) -> int: ...
    @property
    def seconds(self) -> int: ...
    @property
    def remaining_seconds(self) -> int: ...
    @property
    def microseconds(self) -> int: ...
    @property
    def invert(self) -> bool: ...
    def in_weeks(self) -> int: ...
    def in_days(self) -> int: ...
    def in_hours(self) -> int: ...
    def in_minutes(self) -> int: ...
    def in_seconds(self) -> int: ...
    def in_words(self, locale: Optional[str] = ..., separator: str = ...) -> str: ...
    def as_timedelta(self) -> timedelta: ...
    def __add__(self: _Self, other: timedelta) -> _Self: ...
    __radd__: Any = ...
    def __sub__(self: _Self, other: timedelta) -> _Self: ...
    def __neg__(self: _Self) -> _Self: ...
    def __mul__(self: _Self, other: Union[int, float]) -> _Self: ...
    __rmul__ = __mul__
    @overload
    def __floordiv__(self: _Self, other: timedelta) -> int: ...
    @overload
    def __floordiv__(self: _Self, other: int) -> _Self: ...
    @overload
    def __truediv__(self: _Self, other: timedelta) -> float: ...
    @overload
    def __truediv__(self: _Self, other: Union[float, int]) -> _Self: ...
    __div__ = __floordiv__
    def __mod__(self: _Self, other: timedelta) -> _Self: ...
    def __divmod__(self: _Self, other: timedelta) -> Tuple[int, _Self]: ...

class AbsoluteDuration(Duration):
    def __new__(cls, days: int = ..., seconds: int = ..., microseconds: int = ..., milliseconds: int = ..., minutes: int = ..., hours: int = ..., weeks: int = ..., years: int = ..., months: int = ...) -> AbsoluteDuration: ...