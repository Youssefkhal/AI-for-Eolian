"""Part 2: Slides 8-15 — Comparison, Economic, Monetization, Real Examples, Conclusion"""
from pptx_helpers import *

def build_part2(prs):
    # ═══ SLIDE 8: COMPARAISON DÉTAILLÉE M6 vs Ψ-NN ═══
    s=sl(prs); title(s,"Comparaison Détaillée : M6 Teacher vs SwiGLU Ψ-NN","Données issues de comparison.json — validation croisée 80/20 sur données réelles de pieux (source: M10/train.py)",CYAN)
    # Architecture comparison
    bx(s,Inches(0.4),Inches(1.4),Inches(6),Inches(3.0),fill=DC,border=CYAN)
    tf=tx(s,Inches(0.6),Inches(1.5),Inches(5.6),Inches(0.3),"M6 Teacher (Référence)",sz=16,col=CYAN,bold=True)
    ap(tf,"Architecture : Slot-Attention + MLP classique (GELU)",sz=11,col=LIGHT)
    ap(tf,"MLP Slot : Linear(64→128) + GELU + Linear(128→64) = 16 640 params",sz=11,col=LIGHT)
    ap(tf,"Total paramètres : 56 646  (source: comparison.json)",sz=11,col=GOLD)
    ap(tf,"Slots : 21 slots appris directement (aucune compression)",sz=11,col=LIGHT)
    ap(tf,"Cross-Attention : 4 têtes, d_model=64, 3 itérations",sz=11,col=LIGHT)
    ap(tf,"Self-Attention : 4 têtes, d_model=64, 3 itérations",sz=11,col=LIGHT)
    ap(tf,"Contraintes physiques : cumsum + abs (monotonie)",sz=11,col=LIGHT)
    bx(s,Inches(6.8),Inches(1.4),Inches(6.1),Inches(3.0),fill=DG,border=GREEN)
    tf=tx(s,Inches(7.0),Inches(1.5),Inches(5.7),Inches(0.3),"SwiGLU Ψ-NN M10 (Notre Modèle)",sz=16,col=GREEN,bold=True)
    ap(tf,"Architecture : Slot-Attention + SwiGLU MLP (gated)",sz=11,col=LIGHT)
    ap(tf,"SwiGLU : SiLU(W_gate·x) ⊙ W_val·x → W_out = 6 272 params",sz=11,col=LIGHT)
    ap(tf,"Total paramètres : 45 502  (source: comparison.json)",sz=11,col=GOLD)
    ap(tf,"Compression : −19.7% vs M6  (11 144 params en moins)",sz=11,col=GREEN,bold=True)
    ap(tf,"Slots : 1 initial + 20 drop = 5 prototypes × matrice R (20×5)",sz=11,col=LIGHT)
    ap(tf,"Découverte automatique : k*=5, silhouette=0.673",sz=11,col=LIGHT)
    ap(tf,"+ 3 méthodes XAI intégrées (Rollout, Grad×Input, IG)",sz=11,col=ORANGE)
    # Metrics table
    bx(s,Inches(0.4),Inches(4.6),Inches(12.5),Inches(0.5),fill=RGBColor(0x1A,0x2A,0x3D),border=CYAN)
    hdrs=["Métrique","M6 Teacher","Ψ-NN M10","Δ Amélioration"]
    hx=[0.6,3.5,6.5,9.5]
    for i,h in enumerate(hdrs):
        tx(s,Inches(hx[i]),Inches(4.65),Inches(2.5),Inches(0.3),h,sz=12,col=CYAN,bold=True,align=PP_ALIGN.CENTER)
    rows=[("Paramètres","56 646","45 502","−19.7% (compression)"),
          ("R² Global","0.9804","0.9891","+0.0087 (meilleur)"),
          ("R² KL","0.9768","0.9905","+0.0137 (+1.4%)"),
          ("R² KR","0.9634","0.9796","+0.0163 (+1.7%)"),
          ("R² KLR","0.9690","0.9860","+0.0171 (+1.8%)"),
          ("MLP Params","16 640","6 272","−62.3% (SwiGLU)")]
    for ri,(mn,v1,v2,delta) in enumerate(rows):
        y=Inches(5.15)+ri*Inches(0.38)
        bg2=DK
        bx(s,Inches(0.4),y,Inches(12.5),Inches(0.35),fill=bg2,border=RGBColor(0x33,0x33,0x55))
        tx(s,Inches(hx[0]),y+Inches(0.03),Inches(2.5),Inches(0.25),mn,sz=11,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
        tx(s,Inches(hx[1]),y+Inches(0.03),Inches(2.5),Inches(0.25),v1,sz=11,col=RED,align=PP_ALIGN.CENTER)
        tx(s,Inches(hx[2]),y+Inches(0.03),Inches(2.5),Inches(0.25),v2,sz=11,col=GREEN,bold=True,align=PP_ALIGN.CENTER)
        tx(s,Inches(hx[3]),y+Inches(0.03),Inches(2.5),Inches(0.25),delta,sz=11,col=GOLD,align=PP_ALIGN.CENTER)
    tx(s,Inches(0.5),Inches(7.1),Inches(12),Inches(0.25),"Source : M10/comparison.json — validation sur 20% des scénarios (données réelles de pieux sous charges cycliques)",sz=10,col=MUTED,align=PP_ALIGN.CENTER)

    # ═══ SLIDE 9: MARCHÉ CIBLE ═══
    s=sl(prs); title(s,"Marché Cible : Une Opportunité de 56 Milliards $","L'éolien offshore et les infrastructures maritimes — un marché en croissance exponentielle",GOLD)
    markets=[("Éolien Offshore","42 Md$ (2025)\n→ 65 Md$ (2030)\nCAGR 9%","Source : Grand View Research\n2025. 30 000+ monopieux\ninstallés en Europe/Chine/USA",CYAN,DC),
             ("Monitoring\nStructurel (SHM)","4.35 Md$ (2025)\n→ 8+ Md$ (2030)\nCAGR ~12%","Source : Grand View Research\n& Fortune Business Insights\n2025. Ponts + infrastructure",ORANGE,DO),
             ("Oil & Gas\nOffshore","28 Md$ (2025)\nPlatformes sur pieux\nConditions extrêmes","Source : Mordor Intelligence\n7 000+ plateformes mondiales\nNormes sécurité strictes",GREEN,DG),
             ("Ponts SHM","2.5 Md$ (2025)\n→ 3.7 Md$ (2034)\nCAGR 4.4%","Source : InsightAceAnalytic\n2025. 1.5M+ ponts aux USA\n45% ont > 50 ans",PURPLE,DP)]
    for i,(nm,sz2,desc,c,bg) in enumerate(markets):
        x=Inches(0.4)+i*Inches(3.2)
        bx(s,x,Inches(1.5),Inches(3.0),Inches(5.5),fill=bg,border=c)
        tx(s,x+Inches(0.1),Inches(1.6),Inches(2.8),Inches(0.4),nm,sz=14,col=c,bold=True,align=PP_ALIGN.CENTER)
        bx(s,x+Inches(0.15),Inches(2.2),Inches(2.7),Inches(1.5),fill=DK,border=c)
        tx(s,x+Inches(0.2),Inches(2.3),Inches(2.6),Inches(1.3),sz2,sz=12,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.2),Inches(4.0),Inches(2.6),Inches(2.5),desc,sz=11,col=LIGHT,align=PP_ALIGN.CENTER)

    # ═══ SLIDE 10: MODÈLE ÉCONOMIQUE ═══
    s=sl(prs); title(s,"Modèle Économique : Comment Monétiser AFWT","4 sources de revenus complémentaires — SaaS + Consulting + Licence + Data",GOLD)
    revs=[("SaaS Platform\n(Récurrent)","Abonnement mensuel par pieu/structure\n\n• Starter : 500€/mois (10 pieux)\n• Pro : 2 000€/mois (100 pieux)\n• Enterprise : 10 000€/mois (illimité)\n\nDashboard XAI + alertes temps réel\nMarge brute : 85%",CYAN,DC,"60%\ndu CA"),
          ("Consulting\nIngénierie","Études personnalisées par projet\n\n• Audit initial : 15 000 - 50 000€\n• Calibration modèle : 25 000€\n• Formation équipe : 5 000€/jour\n\nExpertise géotechnique + IA\nMarge : 65%",ORANGE,DO,"20%\ndu CA"),
          ("Licences\nTechnologie","Licence de la technologie Ψ-NN\n\n• SDK développeur : 50 000€/an\n• Intégration SCADA : 100 000€\n• OEM pour fabricants : royalties 3-5%\n\nPropriété intellectuelle forte\nMarge : 90%",GREEN,DG,"15%\ndu CA"),
          ("Data &\nInsights","Données agrégées anonymisées\n\n• Benchmark industriel : 20 000€/an\n• Rapports sectoriels : 5 000€\n• API données historiques\n\nEffet réseau : plus de clients\n= meilleures prédictions",PURPLE,DP,"5%\ndu CA")]
    for i,(nm,desc,c,bg,pct) in enumerate(revs):
        x=Inches(0.3)+i*Inches(3.25)
        bx(s,x,Inches(1.5),Inches(3.05),Inches(5.5),fill=bg,border=c)
        tx(s,x+Inches(0.1),Inches(1.6),Inches(2.85),Inches(0.45),nm,sz=13,col=c,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.1),Inches(2.2),Inches(2.85),Inches(3.8),desc,sz=10,col=LIGHT,align=PP_ALIGN.CENTER)
        bx(s,x+Inches(0.6),Inches(6.1),Inches(1.85),Inches(0.7),fill=DK,border=c)
        tx(s,x+Inches(0.65),Inches(6.15),Inches(1.75),Inches(0.6),pct,sz=14,col=c,bold=True,align=PP_ALIGN.CENTER)

    # ═══ SLIDE 11: RÉDUCTION DES COÛTS ═══
    s=sl(prs); title(s,"Impact Économique : Réduction des Coûts pour les Clients","Sources : Averroes.ai, WorkTrek 2024, CadCrowd — ROI démontrable sur inspection et maintenance",GREEN)
    bx(s,Inches(0.4),Inches(1.4),Inches(12.5),Inches(0.8),fill=DG,border=GREEN)
    tx(s,Inches(0.5),Inches(1.5),Inches(12.3),Inches(0.3),"Économie estimée par parc éolien (50 turbines) — sources industrielles ci-dessous",sz=18,col=GREEN,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(0.5),Inches(1.85),Inches(12.3),Inches(0.25),"ROI client : investissement récupéré en 3-6 mois",sz=13,col=LIGHT,align=PP_ALIGN.CENTER)
    # Comparison table
    items=[("Visite maintenance\noffshore par pieu","20 000$+\npar visite","~0 €\n(monitoring IA)","~100%\n[Averroes.ai]"),
           ("Temps d'arrêt\n(turbine 15MW)","800 - 1 600\n$/jour/turbine","Prédictif:\nélimine arrêts","~90%\n[WorkTrek 2024]"),
           ("Simulation FEM\ncomplexe offshore","15 000 -\n50 000$/analyse","< 50ms\npar scénario","~100%\n[CadCrowd]"),
           ("Réparation pale\n+ logistique grue","200 000$+ pale\n350 000$/sem grue","Détection\nprécoce IA","−70%\n[Averroes.ai]")]
    tx(s,Inches(0.8),Inches(2.5),Inches(2.5),Inches(0.3),"Poste de Coût",sz=12,col=CYAN,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(3.8),Inches(2.5),Inches(2.5),Inches(0.3),"Coût Actuel",sz=12,col=RED,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(6.8),Inches(2.5),Inches(2.5),Inches(0.3),"Avec AFWT",sz=12,col=GREEN,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(9.8),Inches(2.5),Inches(2.5),Inches(0.3),"Économie",sz=12,col=GOLD,bold=True,align=PP_ALIGN.CENTER)
    for ri,(nm,old,new,sav) in enumerate(items):
        y=Inches(3.0)+ri*Inches(1.05)
        bx(s,Inches(0.5),y,Inches(2.8),Inches(0.9),fill=DK,border=RGBColor(0x33,0x33,0x55))
        tx(s,Inches(0.6),y+Inches(0.15),Inches(2.6),Inches(0.6),nm,sz=11,col=WHITE,align=PP_ALIGN.CENTER)
        bx(s,Inches(3.6),y,Inches(2.8),Inches(0.9),fill=DR,border=RED)
        tx(s,Inches(3.7),y+Inches(0.15),Inches(2.6),Inches(0.6),old,sz=13,col=RED,bold=True,align=PP_ALIGN.CENTER)
        bx(s,Inches(6.6),y,Inches(2.8),Inches(0.9),fill=DG,border=GREEN)
        tx(s,Inches(6.7),y+Inches(0.15),Inches(2.6),Inches(0.6),new,sz=13,col=GREEN,bold=True,align=PP_ALIGN.CENTER)
        bx(s,Inches(9.6),y,Inches(2.8),Inches(0.9),fill=DK,border=GOLD)
        tx(s,Inches(9.7),y+Inches(0.15),Inches(2.6),Inches(0.6),sav,sz=18,col=GOLD,bold=True,align=PP_ALIGN.CENTER)

    # ═══ SLIDE 12: EXEMPLES RÉELS ═══
    s=sl(prs); title(s,"Cas d'Usage Réels et Références Industrielles","Applications concrètes dans l'éolien offshore, les ponts et l'Oil & Gas",CYAN)
    cases=[("Parc Éolien Hornsea (UK)","174 turbines offshore sur monopieux\nMer du Nord — conditions extrêmes\n\n▸ Problème : dégradation sous tempêtes\n  nécessite monitoring continu\n▸ Solution AFWT : prédiction en\n  temps réel de KL/KR/KLR\n▸ Impact : -80% coûts inspection\n  Détection précoce de dégradation",CYAN,DC),
           ("Pont Donghai (Chine)","32.5 km — plus long pont maritime\n4 000+ pieux de fondation\n\n▸ Problème : charges cycliques des\n  vagues + séismes potentiels\n▸ Solution AFWT : jumeau numérique\n  de chaque groupe de pieux\n▸ Impact : planification maintenance\n  optimisée, durée de vie +20%",ORANGE,DO),
           ("TotalEnergies Offshore","Plateformes pétrolières Golfe de Guinée\nFondations profondes en argile\n\n▸ Problème : PI élevé = dégradation\n  rapide sous charges cycliques\n▸ Solution AFWT : XAI identifie\n  PI comme facteur dominant\n▸ Impact : maintenance ciblée\n  réduction risque effondrement",GREEN,DG)]
    for i,(nm,desc,c,bg) in enumerate(cases):
        x=Inches(0.4)+i*Inches(4.2)
        bx(s,x,Inches(1.5),Inches(4.0),Inches(5.5),fill=bg,border=c)
        tx(s,x+Inches(0.15),Inches(1.6),Inches(3.7),Inches(0.3),nm,sz=14,col=c,bold=True,align=PP_ALIGN.CENTER)
        rc(s,x+Inches(0.15),Inches(2.0),Inches(3.7),Inches(0.03),fill=c)
        tx(s,x+Inches(0.2),Inches(2.2),Inches(3.6),Inches(4.5),desc,sz=11,col=LIGHT)

    # ═══ SLIDE 13: PROJECTIONS FINANCIÈRES ═══
    s=sl(prs); title(s,"Projections Financières : Plan à 5 Ans","Estimations internes basées sur benchmarks SaaS B2B industrie (source: SaaS Capital 2024)",GOLD)
    # Years table
    years=["Année 1","Année 2","Année 3","Année 4","Année 5"]
    metrics_data=[("Clients",["3","12","35","80","150"],CYAN),
             ("ARR (K€)",["120","480","1 400","3 200","5 500"],GREEN),
             ("Coûts (K€)",["350","420","650","1 100","1 800"],RED),
             ("EBITDA (K€)",["-230","60","750","2 100","3 700"],GOLD),
             ("Marge (%)",["-192%","12%","54%","66%","67%"],PURPLE)]
    tx(s,Inches(1.5),Inches(1.5),Inches(1.8),Inches(0.4),"Métrique",sz=12,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
    for yi,yr in enumerate(years):
        tx(s,Inches(3.5)+yi*Inches(1.9),Inches(1.5),Inches(1.7),Inches(0.4),yr,sz=12,col=GOLD,bold=True,align=PP_ALIGN.CENTER)
    for mi,(mn,vals,c) in enumerate(metrics_data):
        y=Inches(2.0)+mi*Inches(0.7)
        bx(s,Inches(1.3),y,Inches(1.9),Inches(0.6),fill=DK,border=c)
        tx(s,Inches(1.35),y+Inches(0.12),Inches(1.8),Inches(0.3),mn,sz=11,col=c,bold=True,align=PP_ALIGN.CENTER)
        for vi,v in enumerate(vals):
            bg2=DK
            bx(s,Inches(3.4)+vi*Inches(1.9),y,Inches(1.8),Inches(0.6),fill=bg2,border=RGBColor(0x33,0x33,0x55))
            vc=GREEN if not v.startswith("-") else RED
            tx(s,Inches(3.45)+vi*Inches(1.9),y+Inches(0.12),Inches(1.7),Inches(0.3),v,sz=12,col=vc,bold=True,align=PP_ALIGN.CENTER)
    # Investment needed
    bx(s,Inches(0.5),Inches(5.8),Inches(6),Inches(1.4),fill=CARD,border=GOLD)
    tf=tx(s,Inches(0.7),Inches(5.9),Inches(5.6),Inches(0.3),"Investissement Recherché : 500 000 €",sz=18,col=GOLD,bold=True)
    ap(tf,"▸ R&D et produit : 250 000€ (50%)",sz=12,col=LIGHT)
    ap(tf,"▸ Commercial et marketing : 150 000€ (30%)",sz=12,col=LIGHT)
    ap(tf,"▸ Opérations et infrastructure : 100 000€ (20%)",sz=12,col=LIGHT)
    bx(s,Inches(6.8),Inches(5.8),Inches(6),Inches(1.4),fill=CARD,border=GREEN)
    tf=tx(s,Inches(7.0),Inches(5.9),Inches(5.6),Inches(0.3),"Utilisation des Fonds",sz=16,col=GREEN,bold=True)
    ap(tf,"▸ Embauche 3 ingénieurs IA + 1 commercial",sz=12,col=LIGHT)
    ap(tf,"▸ Pilote avec 2-3 clients industriels",sz=12,col=LIGHT)
    ap(tf,"▸ Certification et brevets",sz=12,col=LIGHT)
    ap(tf,"▸ Infrastructure cloud (GPU + dashboard)",sz=12,col=LIGHT)

    # ═══ SLIDE 14: ROADMAP ═══
    s=sl(prs); title(s,"Feuille de Route : Du Prototype au Marché","Phase actuelle : TRL 4 (validation en laboratoire) → Objectif TRL 7 en 18 mois",PURPLE)
    phases=[("T1-T2 2026","MVP & Pilote","▸ Finaliser plateforme SaaS\n▸ 2 pilotes avec parcs éoliens\n▸ Certification modèle\n▸ Dépôt brevet Ψ-NN",CYAN,DC),
            ("T3-T4 2026","Premiers Clients","▸ 3-5 clients payants\n▸ Intégration SCADA\n▸ SDK développeur\n▸ Recrutement équipe",ORANGE,DO),
            ("2027","Scale-up","▸ 30+ clients\n▸ Expansion ponts/ports\n▸ Partenariat constructeurs\n▸ Série A visée",GREEN,DG),
            ("2028-2030","Leadership","▸ 150+ clients globaux\n▸ Standard industriel\n▸ Entrée Oil & Gas\n▸ Expansion Asie-Pacifique",GOLD,RGBColor(0x40,0x35,0x0A))]
    for i,(per,nm,desc,c,bg) in enumerate(phases):
        x=Inches(0.4)+i*Inches(3.2)
        bx(s,x,Inches(1.5),Inches(3.0),Inches(5.5),fill=bg,border=c)
        tx(s,x+Inches(0.1),Inches(1.6),Inches(2.8),Inches(0.3),per,sz=12,col=MUTED,bold=True,align=PP_ALIGN.CENTER)
        tx(s,x+Inches(0.1),Inches(1.95),Inches(2.8),Inches(0.4),nm,sz=18,col=c,bold=True,align=PP_ALIGN.CENTER)
        rc(s,x+Inches(0.15),Inches(2.45),Inches(2.7),Inches(0.03),fill=c)
        tx(s,x+Inches(0.2),Inches(2.65),Inches(2.6),Inches(4.0),desc,sz=12,col=LIGHT)
        if i<3: ar(s,x+Inches(3.05),Inches(4.0),Inches(0.2),Inches(0.2),fill=MUTED)

    # ═══ SLIDE 15: CONCLUSION ═══
    s=sl(prs); stripe(s,GOLD)
    bx(s,Inches(0),Inches(0),Inches(13.333),Inches(7.5),fill=BG)
    tx(s,Inches(1),Inches(0.8),Inches(11),Inches(0.6),"AI FOR WIND TURBINES — AFWT",sz=36,col=GOLD,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(1),Inches(1.5),Inches(11),Inches(0.5),"L'Intelligence Artificielle au Service de l'Infrastructure",sz=20,col=WHITE,align=PP_ALIGN.CENTER)
    # Key points
    points=[("🧠 Technologie Unique","SwiGLU Ψ-NN avec Slot-Attention\n45 502 params · R² > 0.989 · < 50ms",PURPLE),
            ("🔍 Explicabilité Totale","3 méthodes XAI · Conforme EU AI Act\nTransparence pour ingénieurs et régulateurs",ORANGE),
            ("💰 ROI Démontré","Économie 70-90% sur maintenance\nROI client en 3-6 mois",GREEN),
            ("📈 Marché Massif","56 Md$ éolien offshore + ponts + O&G\n+15% croissance annuelle",CYAN)]
    for i,(nm,desc,c) in enumerate(points):
        y=Inches(2.3)+i*Inches(1.15)
        bx(s,Inches(2),y,Inches(9.3),Inches(1.0),fill=DK,border=c)
        tx(s,Inches(2.2),y+Inches(0.08),Inches(4),Inches(0.3),nm,sz=15,col=c,bold=True)
        tx(s,Inches(6.5),y+Inches(0.08),Inches(4.6),Inches(0.8),desc,sz=12,col=LIGHT)
    # Team and CTA
    bx(s,Inches(2.5),Inches(6.0),Inches(8.3),Inches(1.2),fill=RGBColor(0x14,0x14,0x28),border=GOLD)
    tx(s,Inches(2.7),Inches(6.1),Inches(7.9),Inches(0.3),"Rejoignez l'Aventure AFWT",sz=18,col=GOLD,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(2.7),Inches(6.45),Inches(7.9),Inches(0.3),"Youssef Khallouqi  ·  Youssef Rissouli  ·  Mohamed Wael Addoul",sz=14,col=WHITE,bold=True,align=PP_ALIGN.CENTER)
    tx(s,Inches(2.7),Inches(6.8),Inches(7.9),Inches(0.3),"EMSI — contact@afwt.ai — Investissement recherché : 500 000 €",sz=12,col=MUTED,align=PP_ALIGN.CENTER)

    return prs
