from src.gateway.training import progress

LOG = """\
### PHASE 1/3 domain=kicad-dsl seq=512
Iter 1: Val loss 3.700, Val took 21.0s
Iter 50: Train loss 2.10, It/sec 0.10
### PHASE 2/3 domain=kicad-dsl seq=1280
Iter 1: Val loss 1.80, Val took 30.0s
Iter 340: Train loss 0.90, It/sec 0.03
"""

DONE_LOG = LOG + "### PHASE 3/3 domain=kicad-dsl seq=2048\nIter 500: Train loss 0.40\n### DOMAIN COMPLETE kicad-dsl final_val_loss=0.412\n"


def test_parse_in_progress():
    p = progress.parse_domain_log(LOG, "kicad-dsl")
    assert p.phase == 2
    assert p.iter == 340
    assert p.complete is False
    assert p.final_val_loss is None


def test_parse_complete():
    p = progress.parse_domain_log(DONE_LOG, "kicad-dsl")
    assert p.complete is True
    assert p.final_val_loss == 0.412


def test_parse_empty_log():
    p = progress.parse_domain_log("", "kicad-dsl")
    assert p.phase == 0 and p.iter == 0 and p.complete is False


def test_classify_val_loss():
    assert progress.classify_val_loss(0.001) == "SUSPECT_OVERFIT"
    assert progress.classify_val_loss(2.0) == "SUSPECT_UNDERTRAIN"
    assert progress.classify_val_loss(0.412) == "OK"
