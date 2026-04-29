"""Main runner: Build complete AFWT Creation d'Entreprise PPTX"""
import os
from pptx_helpers import mk
from make_entreprise_pptx_p1 import build_part1
from make_entreprise_pptx_p2 import build_part2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, "AFWT_Creation_Entreprise.pptx")

prs = mk()
prs = build_part1(prs)
prs = build_part2(prs)
prs.save(OUTPUT)
print(f"Saved: {OUTPUT}")
print(f"Total slides: {len(prs.slides)}")
