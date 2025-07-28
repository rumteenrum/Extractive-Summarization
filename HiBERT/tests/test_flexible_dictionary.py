
from fairseq.data import FlexibleDictionary

import tempfile
import unittest

import torch

from fairseq.data import FlexibleDictionary as Dictionary
from fairseq.tokenizer import Tokenizer


class TestDictionary(unittest.TestCase):

    def test_finalize(self):
        txt = [
            'A B C D E',
            'B C D',
            'C D',
            'D',
        ]
        ref_ids1 = list(map(torch.IntTensor, [
            [4, 5, 6, 7, 8, 2],
            [5, 6, 7, 2],
            [6, 7, 2],
            [7, 2],
        ]))
        ref_ids2 = list(map(torch.IntTensor, [
            [7, 6, 5, 4, 8, 2],
            [6, 5, 4, 2],
            [5, 4, 2],
            [4, 2],
        ]))

        # build dictionary
        d = Dictionary(luaHeritage=True)
        print(d)
        for line in txt:
            ids = Tokenizer.tokenize(line, d, add_if_not_exist=True)
            print(ids)

        print('vocab size', len(d))

        def get_ids(dictionary):
            ids = []
            for line in txt:
                ids.append(Tokenizer.tokenize(line, dictionary, add_if_not_exist=False))
            return ids

        def assertMatch(ids, ref_ids):
            for toks, ref_toks in zip(ids, ref_ids):
                self.assertEqual(toks.size(), ref_toks.size())
                self.assertEqual(0, (toks != ref_toks).sum().item())

        ids = get_ids(d)
        assertMatch(ids, ref_ids1)

        # check finalized dictionary
        d.finalize()
        print('vocab size', len(d))

        finalized_ids = get_ids(d)
        print( finalized_ids )
        assertMatch(finalized_ids, ref_ids2)

        # write to disk and reload
        with tempfile.NamedTemporaryFile(mode='w') as tmp_dict:
            d.save(tmp_dict.name)
            d = Dictionary.load(tmp_dict.name)
            reload_ids = get_ids(d)
            print( reload_ids )
            assertMatch(reload_ids, ref_ids2)
            assertMatch(finalized_ids, reload_ids)


class TestDictionaryNoluaHeritage(unittest.TestCase):

    def test_finalize(self):
        txt = [
            'A B C D E',
            'B C D',
            'C D',
            'D',
        ]
        ref_ids1 = list(map(torch.IntTensor, [
            [4, 5, 6, 7, 8, 2],
            [5, 6, 7, 2],
            [6, 7, 2],
            [7, 2],
        ]))
        ref_ids2 = list(map(torch.IntTensor, [
            [7, 6, 5, 4, 8, 2],
            [6, 5, 4, 2],
            [5, 4, 2],
            [4, 2],
        ]))

        ref_ids1 = [ref_id-1 for ref_id in ref_ids1]
        ref_ids2 = [ref_id-1 for ref_id in ref_ids2]

        # build dictionary
        d = Dictionary(luaHeritage=False)
        print(d)
        for line in txt:
            ids = Tokenizer.tokenize(line, d, add_if_not_exist=True)
            print(ids)

        print('vocab size', len(d))

        def get_ids(dictionary):
            ids = []
            for line in txt:
                ids.append(Tokenizer.tokenize(line, dictionary, add_if_not_exist=False))
            return ids

        def assertMatch(ids, ref_ids):
            for toks, ref_toks in zip(ids, ref_ids):
                self.assertEqual(toks.size(), ref_toks.size())
                self.assertEqual(0, (toks != ref_toks).sum().item())

        ids = get_ids(d)
        assertMatch(ids, ref_ids1)

        # check finalized dictionary
        d.finalize()
        print('vocab size', len(d))

        finalized_ids = get_ids(d)
        print( finalized_ids )
        assertMatch(finalized_ids, ref_ids2)

        # write to disk and reload
        with tempfile.NamedTemporaryFile(mode='w') as tmp_dict:
            d.save(tmp_dict.name)
            d = Dictionary.load(tmp_dict.name)
            reload_ids = get_ids(d)
            print( reload_ids )
            assertMatch(reload_ids, ref_ids2)
            assertMatch(finalized_ids, reload_ids)



class TestDictionaryNoluaHeritageNoEOS(unittest.TestCase):

    def test_finalize(self):
        txt = [
            'A B C D E',
            'B C D',
            'C D',
            'D',
        ]
        ref_ids1 = list(map(torch.IntTensor, [
            [4, 5, 6, 7, 8, 2],
            [5, 6, 7, 2],
            [6, 7, 2],
            [7, 2],
        ]))
        ref_ids2 = list(map(torch.IntTensor, [
            [7, 6, 5, 4, 8, 2],
            [6, 5, 4, 2],
            [5, 4, 2],
            [4, 2],
        ]))

        ref_ids1 = [ref_id-2 for ref_id in ref_ids1]
        ref_ids2 = [ref_id-2 for ref_id in ref_ids2]

        # build dictionary
        d = Dictionary([('EOS', '</s>'), ('UNK', '<unk>')], luaHeritage=False)
        print(d)
        for line in txt:
            ids = Tokenizer.tokenize(line, d, add_if_not_exist=True)
            print(ids)

        print('vocab size', len(d))

        def get_ids(dictionary):
            ids = []
            for line in txt:
                ids.append(Tokenizer.tokenize(line, dictionary, add_if_not_exist=False))
            return ids

        def assertMatch(ids, ref_ids):
            for toks, ref_toks in zip(ids, ref_ids):
                self.assertEqual(toks.size(), ref_toks.size())
                self.assertEqual(0, (toks != ref_toks).sum().item())

        ids = get_ids(d)
        assertMatch(ids, ref_ids1)

        # check finalized dictionary
        d.finalize()
        print('vocab size', len(d))

        finalized_ids = get_ids(d)
        print( finalized_ids )
        assertMatch(finalized_ids, ref_ids2)

        # write to disk and reload
        with tempfile.NamedTemporaryFile(mode='w') as tmp_dict:
            d.save(tmp_dict.name)
            d = Dictionary.load(tmp_dict.name)
            reload_ids = get_ids(d)
            print( reload_ids )
            assertMatch(reload_ids, ref_ids2)
            assertMatch(finalized_ids, reload_ids)


if __name__ == '__main__':
    unittest.main()
