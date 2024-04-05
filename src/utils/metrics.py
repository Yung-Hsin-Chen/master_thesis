import jiwer
# from datasets import load_metric
# from evaluate import load

# def compute_cer(pred_ids, label_ids):
#     cer_metric = load_metric("cer")
#     cer_score = cer_metric.compute(predictions=pred_ids, references=label_ids)
#     wer = load("wer")
#     wer_score = wer.compute(predictions=pred_ids, references=label_ids)
#     return cer_score, wer_score


def get_wer_cer_per_batch(targets, labels):
    wer_score, cer_score = 0, 0
    # print("TARGET: ", targets)
    # print("LABEL:", labels)
    # targets = get_word(ocr_output)
    # labels = get_word(label)
    # targets = [convert_string(s) for s in targets]
    # labels = [convert_string(s) for s in labels]
    for t,l in zip(targets, labels):
        if not t:
            wer_score += 1
            cer_score += 1
            return wer_score, cer_score
        # print("OUTPUT: ", t)
        # print("LABEL: ", l)
        wer_score += jiwer.wer(t, l)
        cer_score += jiwer.cer(t, l)
    # for t,l in zip(targets, labels):
    #     if t.strip()=="":
    #         wer_score += 1
    #         cer_score += 1
    #     else:
    #         wer_score += jiwer.wer(t, l)
    #         cer_score += jiwer.cer(t, l)
    return wer_score, cer_score
