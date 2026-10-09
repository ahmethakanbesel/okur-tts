"""What Okur reads for the demo page's default sentence and examples, so the page shows the reading before the model
loads. Rewrites web/src/readings.json from the texts in web/src/i18n.ts.

    uv run python scripts/web_readings.py
"""

import json
import re
from pathlib import Path

from okur.frontend import Frontend

source = Path("web/src/i18n.ts").read_text(encoding="utf-8")
texts = sorted(set(re.findall(r'"input\.default": "([^"]+)"', source) + re.findall(r'text: "([^"]+)" \}', source)))
frontend = Frontend()
out = {text: frontend(text) for text in texts}
Path("web/src/readings.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(out), "readings")
