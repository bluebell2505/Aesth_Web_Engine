from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class BrandDNA:
    palette: List[str]            # hex colors, e.g. ["#ff0000", ...]
    typography: Dict[str, Any]    # e.g. {"primary":"sans", "secondary":"serif", "notes": "..."}
    spacing: str                  # "compact" | "balanced" | "airy"
    density_score: float          # numeric
    tone: str                     # "warm"|"cool"|"pastel"|"neon"|"neutral"
    vibe_tags: List[str]          # e.g. ["playful","premium"]
    raw: Dict[str, Any] = None    # raw analysis output
