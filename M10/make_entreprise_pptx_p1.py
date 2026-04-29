"""Part 1: Slides 1-7 of Creation d'Entreprise PPTX"""
from pptx_helpers import *
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def build_part1(prs):
    # ═══ SLIDE 1: TITLE ═══
    s=sl(prs); stripe(s,GOLD)
    bx(s,Inches(0),Inches(0),Inches(13.333),Inches(7.5),fill=BG)
    tx(s,Inches(1),Inches(0.8),Inches(11),Inches(0.8),"AI FOR WIND TURBINES — AFWT",sz=40,col=GOLD,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(1),Inches(1.7),Inches(11),Inches(0.6),"Jumeau Numérique Intelligent pour la Prédiction de",sz=22,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(1),Inches(2.2),Inches(11),Inches(0.6),"Dégradation de Rigidité des Pieux sous Charges Cycliques",sz=22,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(1),Inches(3.1),Inches(11),Inches(0.4),"IA Explicable (XAI) · Slot-Attention · SwiGLU · Physics-Informed",sz=16,col=CYAN,align=PP_ALIGN.CENTER)
    # Team
    bx(s,Inches(2.5),Inches(4.2),Inches(8.3),Inches(1.2),fill=RGBColor(0x14,0x14,0x28),border=GOLD)
    tx(s,Inches(2.7),Inches(4.3),Inches(7.9),Inches(0.3),"Équipe Fondatrice",sz=14,col=GOLD,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(2.7),Inches(4.65),Inches(7.9),Inches(0.3),"Youssef Khallouqi  ·  Youssef Rissouli  ·  Mohamed Wael Addoul",sz=16,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(2.7),Inches(5.0),Inches(7.9),Inches(0.3),"EMSI — École Marocaine des Sciences de l'Ingénieur",sz=12,col=MUTED,align=PP_ALIGN.CENTER)
    # Bottom
    tx(s,Inches(1),Inches(5.8),Inches(11),Inches(0.3),"Projet de Création d'Entreprise — Présentation Investisseurs",sz=14,col=PURPLE,align=PP_ALIGN.CENTER)
    tx(s,Inches(1),Inches(6.3),Inches(11),Inches(0.3),"Avril 2026",sz=12,col=MUTED,align=PP_ALIGN.CENTER)
    # Cards
    for i,(lab,ic) in enumerate([("R² > 0.989",CYAN),("19.7% Compression",GREEN),("3 Méthodes XAI",ORANGE),("< 50ms Inférence",PURPLE)]):
        x=Inches(1.5)+i*Inches(2.7)
        bx(s,x,Inches(6.7),Inches(2.3),Inches(0.55),fill=DK,border=ic)
        tx(s,x+Inches(0.1),Inches(6.77),Inches(2.1),Inches(0.3),lab,sz=13,col=ic,bold=True,align=PP_ALIGN.CENTER)

    # ═══ SLIDE 2: PROBLÉMATIQUE ═══
    s=sl(prs); title(s,"Le Problème : Pourquoi AFWT ?","Les fondations de pieux sous charges cycliques — un défi majeur du génie civil",RED)
    # Left
    bx(s,Inches(0.4),Inches(1.4),Inches(6),Inches(5.5),fill=CARD,border=RGBColor(0x33,0x33,0x55))
    tf=tx(s,Inches(0.7),Inches(1.6),Inches(5.4),Inches(0.3),"Contexte Industriel",sz=18,col=RED,bold=True)
    for t in ["Les éoliennes offshore reposent sur des monopieux de 6-10m de diamètre","Les charges cycliques (tempêtes, vagues) dégradent la rigidité du sol","La rigidité latérale (KL), rotationnelle (KR) et couplée (KLR) diminue","Sans monitoring, risque d'effondrement structurel catastrophique"]:
        ap(tf,f"▸ {t}",sz=11,col=LIGHT)
    ap(tf,"",sz=6)
    ap(tf,"Le Coût du Problème",sz=16,col=ORANGE,bold=True)
    for t in ["Inspections physiques : 50 000 - 200 000 € par pieu","Arrêt de production : 5 000 - 15 000 €/jour/éolienne","Remplacement de fondation : 2 - 5 M€","Marché éolien offshore mondial : 56 Md$ en 2025"]:
        ap(tf,f"▸ {t}",sz=11,col=LIGHT)
    # Right
    bx(s,Inches(6.8),Inches(1.4),Inches(6.1),Inches(5.5),fill=CARD,border=RGBColor(0x33,0x33,0x55))
    tf=tx(s,Inches(7.1),Inches(1.6),Inches(5.5),Inches(0.3),"Limites des Solutions Actuelles",sz=18,col=RED,bold=True)
    probs=[("Méthodes Empiriques (p-y)","Imprécises pour charges cycliques complexes, pas de prédiction temporelle",RED),
           ("Éléments Finis (FEM)","Très coûteux : 2-8h par simulation, nécessite expertise pointue",ORANGE),
           ("Monitoring Capteurs Seuls","Données brutes sans prédiction, réactif au lieu de prédictif",YELLOW),
           ("IA Classique (MLP/LSTM)","Boîte noire, pas de garanties physiques, non explicable",PURPLE)]
    for j,(nm,desc,c) in enumerate(probs):
        y=Inches(2.2)+j*Inches(1.2)
        bx(s,Inches(7.0),y,Inches(5.7),Inches(1.0),fill=DK,border=c)
        tx(s,Inches(7.2),y+Inches(0.08),Inches(5.3),Inches(0.25),f"✗ {nm}",sz=12,col=c,bold=True)
        tx(s,Inches(7.2),y+Inches(0.4),Inches(5.3),Inches(0.5),desc,sz=10,col=MUTED)

    # ═══ SLIDE 3: SOLUTION ═══
    s=sl(prs); title(s,"Notre Solution : AFWT","Un jumeau numérique intelligent, explicable et conforme aux lois de la physique",GREEN)
    bx(s,Inches(0.4),Inches(1.4),Inches(12.5),Inches(1.5),fill=DG,border=GREEN)
    tx(s,Inches(0.7),Inches(1.5),Inches(12),Inches(0.4),"AFWT : Prédiction en temps réel + Explicabilité complète",sz=20,col=GREEN,bold=True,align=PP_ALIGN.CENTER)
    tf=tx(s,Inches(0.7),Inches(2.0),Inches(12),Inches(0.3),"8 paramètres sol/pieu → Modèle SwiGLU Ψ-NN → 21 étapes de dégradation KL, KR, KLR + Explications XAI",sz=14,col=LIGHT,align=PP_ALIGN.CENTER)
    cards=[("⚡ Temps Réel","Prédiction < 50ms\nvs 2-8h en FEM\n= 100 000x plus rapide",CYAN,DC),
           ("🔬 Explicable","3 méthodes XAI\nidentifient quels\nparamètres comptent",ORANGE,DO),
           ("⚙️ Physique","Contraintes monotones\nKL↓ KR↓ KLR↑\ngaranties par design",GREEN,DG),
           ("📊 Précis","R² > 0.989\nMAPE < 5%\nsur données réelles",PURPLE,DP)]
    for i,(ic,desc,c,bg) in enumerate(cards):
        x=Inches(0.5)+i*Inches(3.15)
        bx(s,x,Inches(3.2),Inches(2.9),Inches(2.2),fill=bg,border=c)
        tx(s,x+Inches(0.1),Inches(3.3),Inches(2.7),Inches(0.35),ic,sz=14,col=c,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.1),Inches(3.75),Inches(2.7),Inches(1.5),desc,sz=12,col=LIGHT,align=PP_ALIGN.CENTER)
    bx(s,Inches(0.4),Inches(5.7),Inches(12.5),Inches(1.5),fill=CARD,border=RGBColor(0x33,0x33,0x55))
    tf=tx(s,Inches(0.7),Inches(5.8),Inches(12),Inches(0.3),"Avantage Concurrentiel Clé",sz=16,col=GOLD,bold=True)
    ap(tf,"▸ Seule solution combinant : précision FEM + vitesse temps réel + explicabilité complète + garanties physiques",sz=12,col=LIGHT)
    ap(tf,"▸ Architecture brevetable : SwiGLU Ψ-NN — 45 502 params vs 56 646 (M6) = −19.7% (source: comparison.json)",sz=12,col=LIGHT)
    ap(tf,"▸ Marché éolien offshore : 42 Md$ en 2025 → 65 Md$ en 2030 (source: Grand View Research 2025)",sz=12,col=LIGHT)

    # ═══ SLIDE 4: ARCHITECTURE IA ═══
    s=sl(prs); title(s,"Architecture IA : SwiGLU Ψ-NN en Détail","Pipeline 3 étapes : Distillation → Découverte de Structure → Entraînement Ψ-Modèle",PURPLE)
    # Input
    bx(s,Inches(0.3),Inches(1.5),Inches(2.2),Inches(5.5),fill=DC,border=CYAN)
    tx(s,Inches(0.4),Inches(1.6),Inches(2),Inches(0.3),"ENTRÉES",sz=14,col=CYAN,bold=True,align=PP_ALIGN.CENTER)
    feats=["PI","Gmax","v","Dp","Tp","Lp","Ip","Dp/Lp"]
    fcolors=[RED,ORANGE,YELLOW,CYAN,BLUE,GREEN,TEAL,PINK]
    for fi,(f,fc) in enumerate(zip(feats,fcolors)):
        y=Inches(2.0)+fi*Inches(0.58)
        bx(s,Inches(0.45),y,Inches(1.9),Inches(0.45),fill=DK,border=fc)
        tx(s,Inches(0.55),y+Inches(0.08),Inches(1.7),Inches(0.25),f,sz=11,col=fc,bold=True,align=PP_ALIGN.CENTER)
    ar(s,Inches(2.6),Inches(4.0),Inches(0.4),Inches(0.3),fill=CYAN)
    # Model core
    bx(s,Inches(3.2),Inches(1.5),Inches(5.5),Inches(5.5),fill=DP,border=PURPLE)
    tx(s,Inches(3.3),Inches(1.6),Inches(5.3),Inches(0.3),"SwiGLU Ψ-NN (45 502 params)",sz=14,col=PURPLE,bold=True,align=PP_ALIGN.CENTER)
    steps=[("Embedding","Linear(8→64) + LayerNorm + GELU",CYAN),
           ("21 Slots","1 initial + 20 drops (5 prototypes)",ORANGE),
           ("Cross-Attention ×3","Slots ← Input (4 têtes, d=64)",YELLOW),
           ("Self-Attention ×3","Slots ↔ Slots (interactions)",GREEN),
           ("SwiGLU MLP ×3","SiLU(W_gate·x) ⊙ W_val·x",TEAL),
           ("Têtes de Prédiction","Slot → KL, KR, KLR",PINK),
           ("Contraintes Physiques","cumsum + monotonie KL↓ KR↓ KLR↑",RED)]
    for si,(nm,desc,c) in enumerate(steps):
        y=Inches(2.0)+si*Inches(0.7)
        bx(s,Inches(3.4),y,Inches(5.1),Inches(0.58),fill=DK,border=c)
        tx(s,Inches(3.55),y+Inches(0.04),Inches(2.3),Inches(0.22),nm,sz=10,col=c,bold=True)
        tx(s,Inches(5.8),y+Inches(0.04),Inches(2.6),Inches(0.22),desc,sz=9,col=MUTED)
        if si<6: da(s,Inches(5.9),y+Inches(0.55),Inches(0.15),Inches(0.15),fill=RGBColor(0x44,0x44,0x66))
    ar(s,Inches(8.85),Inches(4.0),Inches(0.4),Inches(0.3),fill=PURPLE)
    # Output
    bx(s,Inches(9.4),Inches(1.5),Inches(3.6),Inches(5.5),fill=DG,border=GREEN)
    tx(s,Inches(9.5),Inches(1.6),Inches(3.4),Inches(0.3),"SORTIES",sz=14,col=GREEN,bold=True,align=PP_ALIGN.CENTER)
    for vi,(vn,vd,vc) in enumerate([("KL","Rigidité Latérale\n21 valeurs décroissantes",CYAN),("KR","Rigidité Rotationnelle\n21 valeurs décroissantes",ORANGE),("KLR","Rigidité Couplée\n21 valeurs croissantes",GREEN)]):
        y=Inches(2.1)+vi*Inches(1.6)
        bx(s,Inches(9.6),y,Inches(3.2),Inches(1.4),fill=DK,border=vc)
        tx(s,Inches(9.7),y+Inches(0.1),Inches(1),Inches(0.3),vn,sz=18,col=vc,bold=True)
        tx(s,Inches(9.7),y+Inches(0.5),Inches(3),Inches(0.8),vd,sz=10,col=LIGHT)

    # ═══ SLIDE 5: PIPELINE 3 ÉTAPES ═══
    s=sl(prs); title(s,"Pipeline d'Entraînement en 3 Étapes","De M6 Teacher (56 646 params) à Ψ-Model compressé (45 502 params) — réduction de 19.7%",ORANGE)
    stages=[("ÉTAPE A","Distillation","Le modèle étudiant (SwiGLU)\napprend du professeur M6\navec régularisation L1","Professeur M6 (gelé)\n→ Étudiant SwiGLU\nLoss = L_distill + 0.5·L_data + μ·L1",CYAN,DC),
            ("ÉTAPE B","Découverte de Structure","Clustering K-Means des\n20 slots de dégradation\n→ k*=5 prototypes optimaux","Silhouette Score = 0.673\n20 slots → 5 prototypes\nMatrice R apprise (20×5)",ORANGE,DO),
            ("ÉTAPE C","Entraînement Ψ-Model","Modèle structuré avec\nprototypes + SwiGLU MLP\n+ contraintes physiques","L = L_distill + L_seq + 5·L_init\n+ L_shape + 0.02·L_entropy\n+ 0.2·L_physics_mono",GREEN,DG)]
    for i,(nm,tit,desc,details,c,bg) in enumerate(stages):
        x=Inches(0.5)+i*Inches(4.2)
        bx(s,x,Inches(1.5),Inches(3.9),Inches(5.5),fill=bg,border=c)
        tx(s,x+Inches(0.1),Inches(1.6),Inches(3.7),Inches(0.3),nm,sz=12,col=c,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.1),Inches(1.95),Inches(3.7),Inches(0.3),tit,sz=18,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
        bx(s,x+Inches(0.15),Inches(2.5),Inches(3.6),Inches(1.8),fill=DK,border=c)
        tx(s,x+Inches(0.25),Inches(2.6),Inches(3.4),Inches(1.6),desc,sz=12,col=LIGHT,align=PP_ALIGN.CENTER)
        bx(s,x+Inches(0.15),Inches(4.5),Inches(3.6),Inches(2.2),fill=DK,border=RGBColor(0x33,0x33,0x55))
        tx(s,x+Inches(0.25),Inches(4.55),Inches(3.4),Inches(0.2),"Détails Techniques",sz=10,col=c,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.25),Inches(4.85),Inches(3.4),Inches(1.7),details,sz=10,col=MUTED,align=PP_ALIGN.CENTER,font="Consolas")
        if i<2: ar(s,x+Inches(3.95),Inches(4.0),Inches(0.25),Inches(0.25),fill=MUTED)

    # ═══ SLIDE 6: XAI ═══
    s=sl(prs); title(s,"Explicabilité (XAI) : 3 Méthodes Complémentaires","Transparence totale : du slot au paramètre d'entrée",ORANGE)
    methods=[("Attention Rollout\n(Cross + Self)","Quels slots sont actifs ?\nComment les slots communiquent ?","Cross: moyenne sur 3 itérations\n→ importance par slot [21]\n\nSelf: produit matriciel avec\nmixage résiduel (50%)\n→ matrice de flux [21×21]",CYAN,DC),
             ("Gradient × Input\n(LRP)","Quels paramètres sol/pieu\ninfluencent la prédiction ?","attr_j = x_j × ∂output/∂x_j\n\nSatisfait l'axiome de\ncomplétude du 1er ordre\n\nTemps: ~50ms par scénario",ORANGE,DO),
             ("Gradients Intégrés\n(Gold Standard)","Attribution exacte vers\nles 8 paramètres d'entrée","∫₀¹ ∂f/∂x (baseline + α·Δx) dα\n\nSatisfait exactement:\nf(x) - f(0) = Σ attribution_j\n\nTemps: ~2s (30 pas intégration)",GREEN,DG)]
    for i,(nm,question,how,c,bg) in enumerate(methods):
        x=Inches(0.5)+i*Inches(4.2)
        bx(s,x,Inches(1.5),Inches(3.9),Inches(5.5),fill=bg,border=c)
        tx(s,x+Inches(0.1),Inches(1.6),Inches(3.7),Inches(0.5),nm,sz=14,col=c,bold=True,align=PP_ALIGN.CENTER)
        bx(s,x+Inches(0.15),Inches(2.3),Inches(3.6),Inches(1.0),fill=DK,border=RGBColor(0x33,0x33,0x55))
        tx(s,x+Inches(0.2),Inches(2.3),Inches(3.5),Inches(0.2),"QUESTION:",sz=8,col=MUTED,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.2),Inches(2.55),Inches(3.5),Inches(0.7),question,sz=12,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.2),Inches(3.5),Inches(3.5),Inches(0.2),"MÉTHODE:",sz=8,col=MUTED,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.2),Inches(3.8),Inches(3.5),Inches(2.5),how,sz=10,col=LIGHT,align=PP_ALIGN.CENTER,font="Consolas")

    # ═══ SLIDE 7: PERFORMANCES ═══
    s=sl(prs); title(s,"Performances du Modèle : Résultats Validés","Comparaison M6 Teacher vs Ψ-Model SwiGLU sur données réelles de pieux",CYAN)
    # Table header
    bx(s,Inches(0.5),Inches(1.5),Inches(12.3),Inches(0.6),fill=RGBColor(0x1A,0x2A,0x3D),border=CYAN)
    cols=["Modèle","Params","R² Global","R² KL","R² KR","R² KLR","Compression"]
    xpos=[0.6,3.2,5.0,6.8,8.2,9.6,11.0]
    for ci,cn in enumerate(cols):
        tx(s,Inches(xpos[ci]),Inches(1.55),Inches(1.5),Inches(0.3),cn,sz=11,col=CYAN,bold=True,align=PP_ALIGN.CENTER)
    rows=[("M6 Teacher","56 646","0.9804","0.9768","0.9634","0.9690","—",LIGHT),
          ("Stage-A Student","46 342","0.9885","0.9876","0.9786","0.9827","18.2%",LIGHT),
          ("Ψ-Model SwiGLU","45 502","0.9891","0.9905","0.9796","0.9860","19.7%",GREEN)]
    for ri,(a,b,c2,d,e,f,g,rc2) in enumerate(rows):
        y=Inches(2.2)+ri*Inches(0.55)
        bg2=DK if ri<2 else DG
        bx(s,Inches(0.5),y,Inches(12.3),Inches(0.5),fill=bg2,border=RGBColor(0x33,0x33,0x55) if ri<2 else GREEN)
        vals=[a,b,c2,d,e,f,g]
        for ci,v in enumerate(vals):
            tx(s,Inches(xpos[ci]),y+Inches(0.08),Inches(1.5),Inches(0.3),v,sz=11,col=rc2,bold=(ri==2),align=PP_ALIGN.CENTER)
    # Bottom insights
    bx(s,Inches(0.5),Inches(4.2),Inches(6),Inches(2.8),fill=CARD,border=RGBColor(0x33,0x33,0x55))
    tf=tx(s,Inches(0.7),Inches(4.3),Inches(5.6),Inches(0.3),"Points Clés (source: comparison.json)",sz=16,col=GOLD,bold=True)
    ap(tf,"▸ Ψ-Model : 45 502 params vs M6 : 56 646 = −19.7%",sz=12,col=LIGHT)
    ap(tf,"▸ R² global Ψ : 0.9891 vs M6 : 0.9804 (+0.0087)",sz=12,col=LIGHT)
    ap(tf,"▸ SwiGLU MLP : 6 272 params vs MLP GELU : 16 640 (−62.3%)",sz=12,col=LIGHT)
    ap(tf,"▸ 5 prototypes (silhouette=0.673, source: psi_discovery.json)",sz=12,col=LIGHT)
    ap(tf,"▸ Inférence < 50ms vs 2-8h FEM (source: CadCrowd)",sz=12,col=LIGHT)
    bx(s,Inches(6.8),Inches(4.2),Inches(6),Inches(2.8),fill=CARD,border=RGBColor(0x33,0x33,0x55))
    tf=tx(s,Inches(7.0),Inches(4.3),Inches(5.6),Inches(0.3),"Validation & Sources",sz=16,col=GOLD,bold=True)
    ap(tf,"▸ Données réelles de pieux sous charges cycliques",sz=12,col=LIGHT)
    ap(tf,"▸ Métriques : R² courbe, MAPE, NRMSE par variable",sz=12,col=LIGHT)
    ap(tf,"▸ Contraintes physiques : KL↓ KR↓ KLR↑ (train.py L289-291)",sz=12,col=LIGHT)
    ap(tf,"▸ Cross-validation 80/20 (train.py L754)",sz=12,col=LIGHT)
    ap(tf,"▸ Code reproductible : Python 3.12 / PyTorch",sz=12,col=LIGHT)

    return prs
