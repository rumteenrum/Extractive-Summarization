
import os

def bleu_eval(sys_file, ref_file):
    bleu_file = '%s/multi-bleu.perl' % os.path.dirname(os.path.realpath(__file__))
    assert os.path.exists(bleu_file), 'multi-bleu.perl is missing!!!'
    os.system('perl %s %s < %s'%(bleu_file, ref_file, sys_file))

from .sentence_bleu import SentenceBleuScorer
from nltk.translate.gleu_score import sentence_gleu

# bleu_opt can be "bleu" or "gleu"
def bleu_eval_sent_level(sys_output, reference, spec='n=4', bleu_opt='bleu'):
    if isinstance(sys_output, str):
        sys_output = sys_output.strip().split(' ')
    if isinstance(reference, str):
        reference = reference.strip().split(' ')
    if bleu_opt == 'bleu':
        scorer = SentenceBleuScorer(spec)
        scorer.set_reference(reference)

        return scorer.score(sys_output)
    elif bleu_opt == 'gleu':
        return sentence_gleu([reference], sys_output)
    else:
        raise Exception('invalid bleu opt %s'%bleu_opt)
