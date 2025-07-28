
def get_prec_recall_f1(predict_file, gold_file, show=False):
    def load(infile):
        return [line.strip().split() for line in open(infile, encoding='utf8')]

    predicts = load(predict_file)
    golds = load(gold_file)
    exact_match = 0
    n_seq = len(golds)
    pred_tot = 0
    gold_tot = 0
    correct_tot = 0
    correct_lbl_tot = 0
    lbl_tot = 0
    for i in range(n_seq):
        predict = predicts[i]
        gold = golds[i]
        assert len(predict) == len(gold), 'MUST have the same length'
        seqlen = len(predict)
        pred_cnt, gold_cnt, correct_cnt = 0, 0, 0
        same_cnt = 0
        for j in range(seqlen):
            lbl_pred = predict[j] == 'T'
            lbl_gold = gold[j] == 'T'
            if predict[j] == gold[j]:
                same_cnt += 1
            if lbl_pred:
                pred_cnt += 1
            if lbl_gold:
                gold_cnt += 1
            if lbl_pred and lbl_gold:
                correct_cnt += 1
            if predict[j] == gold[j]:
                correct_lbl_tot += 1
            lbl_tot += 1
        '''
        if correct_cnt == gold_cnt:
            exact_match += 1
        '''
        if same_cnt == seqlen:
            exact_match += 1
        pred_tot += pred_cnt
        gold_tot += gold_cnt
        correct_tot += correct_cnt

    prec = float(correct_tot) / pred_tot if pred_tot > 0 else 0
    recall = float(correct_tot) / gold_tot if gold_tot > 0 else 0
    f1 = 2*prec*recall / (prec+recall) if prec+recall > 0 else 0
    em = float(exact_match) / n_seq
    acc = float(correct_lbl_tot) / lbl_tot
    if show:
        print('prec\t%d / %d = %f'%(correct_tot, pred_tot, prec))
        print('recall\t%d / %d = %f'%(correct_tot, gold_tot, recall))
        print('f1-score\t%f'%f1)
        print('exact match\t%d / %d = %f'%(exact_match, n_seq, em))
        print('lbl acc \t%d / %d = %f'%(correct_lbl_tot, lbl_tot, acc))
        print('\n')

    metrics = dict(prec=prec, recall=recall, f1=f1, exact_match=em, acc=acc)

    return metrics
