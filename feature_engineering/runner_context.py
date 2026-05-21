"""
Runner Context
Contains personalization inputs for feature extraction
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunnerContext:
    """Personal runner characteristics used for feature normalization"""

    age: int  # Runner's age in years
    sex: str  # 'M' or 'F'
    max_hr: int  # Maximum heart rate (bpm)

    # Optional metrics
    experience_years: Optional[int] = None  # Years of running experience
    resting_hr: Optional[int] = None  # Resting heart rate (bpm)
    recent_injury_flag: bool = False  # Has had injury in past 3 months

    def __post_init__(self):
        """Validate inputs"""
        if self.sex not in ['M', 'F']:
            raise ValueError("sex must be 'M' or 'F'")

        if self.max_hr < 100 or self.max_hr > 220:
            raise ValueError("max_hr must be between 100 and 220")

        if self.age < 10 or self.age > 100:
            raise ValueError("age must be between 10 and 100")

        if self.resting_hr is not None and (self.resting_hr < 30 or self.resting_hr > 100):
            raise ValueError("resting_hr must be between 30 and 100")
