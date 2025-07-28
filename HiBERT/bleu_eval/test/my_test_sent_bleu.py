#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bleu_eval import bleu_eval_sent_level

ref = "A A A"
sys = "B B B"

print(bleu_eval_sent_level(sys, ref))

ref = "The very nice man"
sys = "man man man man"

print(bleu_eval_sent_level(sys, ref))


ref = "The very nice man".split(' ')
sys = "man man man man".split(' ')

print(bleu_eval_sent_level(sys, ref))
