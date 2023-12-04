import jiwer

def get_wer_cer_per_batch(ocr_output: torch.Tensor, label: torch.Tensor):
    wer_score, cer_score = 0, 0
    targets = get_word(ocr_output)
    labels = get_word(label)
    targets = [convert_string(s) for s in targets]
    labels = [convert_string(s) for s in labels]
    for t,l in zip(targets, labels):
        if t.strip()=="":
            wer_score += 1
            cer_score += 1
        else:
            wer_score += jiwer.wer(t, l)
            cer_score += jiwer.cer(t, l)
    return wer_score, cer_score
