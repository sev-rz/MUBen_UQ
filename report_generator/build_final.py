from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
from final_r1 import report1
from final_r2 import report2
for n,f in ((1,report1),(2,report2)):
    (HERE/f'Report{n}.html').write_text(f(),encoding='utf-8')
    subprocess.run(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--headless=new','--disable-gpu','--no-pdf-header-footer','--virtual-time-budget=8000',f'--print-to-pdf={HERE}/Report{n}.pdf',f'file://{HERE}/Report{n}.html'],check=True,capture_output=True)
print('ok')
