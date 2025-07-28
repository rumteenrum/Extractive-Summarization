
def get_compression_rate(predict_file):
    def load(infile):
        return [line.strip().split() for line in open(infile, encoding='utf8')]
    predicts = load(predict_file)
    output_length = 0
    length = 0
    for predict in predicts:
        for lbl in predict:
            if lbl == 'T':
                output_length += 1
            length += 1
    return float(output_length) / length

def get_analyze_file(orig_file, gold_file, pred_file, ana_file):
    def load(infile):
        return [line.strip().split() for line in open(infile, encoding='utf8')]

    origs = load(orig_file)
    golds = load(gold_file)
    preds = load(pred_file)
    assert len(origs) == len(golds) and len(golds) == len(preds), 'MUST have the same number of lines'

    def labels2words(orig, lbls):
        words = []
        for word, lbl in zip(orig, lbls):
            if lbl == 'T':
                words.append(word)
        return words

    fout = open(ana_file, 'w', encoding='utf8')
    cnt = 0
    for orig, gold, pred in zip(origs, golds, preds):
        assert len(orig) == len(gold) and len(gold) == len(pred), 'MUST have the same number of words'
        gold_words = labels2words(orig, gold)
        pred_words = labels2words(orig, pred)
        cnt += 1
        fout.write('id = %d\n'%cnt)
        fout.write('[orig] = %s\n'%(' '.join(orig)))
        fout.write('[gold] = %s\n'%(' '.join(gold_words)))
        fout.write('[pred] = %s\n'%(' '.join(pred_words)))
        fout.write('[glbl] = %s\n'%(' '.join(gold)))
        fout.write('[plbl] = %s\n'%(' '.join(pred)))
        fout.write('\n\n\n')
        fout.flush()
    fout.close()
