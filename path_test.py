from pathlib import Path
import os

dir = Path(os.getcwd())
true_path = dir / 'mechanical_turk' / 'HTML_files' / 'HTML_DME'
print(true_path)
open(true_path / 'DME_text_1.html')