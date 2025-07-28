## HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization

## Installation
You need to install python3 and the following libraries.
```
pip install pytorch==1.0
pip install nltk
pip install pyrouge==0.1.3
pip install https://github.com/rsennrich/subword-nmt/archive/master.zip
pip install -r requirements.txt
python setup.py build
python setup.py develop

# dependencies for ROUGE-1.5.5.pl
sudo apt-get update
sudo apt-get install expat
sudo apt-get install libexpat-dev -y

sudo cpan install XML::Parser
sudo cpan install XML::Parser::PerlSAX
sudo cpan install XML::DOM
```

## CNN/Dailmail Dataset
You could download the CNN/Dailymail dataset from https://github.com/abisee/cnn-dailymail and convert it to the `Raw Text Format` and then to the binary format as described below or you could simply contact the authors for the datasets after pre-processing.

### Dataset Raw Text Format
The text format CNN/Dailymail dataset includes the `training`, `validation` and `test` splits and each split contains the `*.article`, `*.summary` and `*.label` files.

Each line in `*.article` is an article and sentences are seperated by `<S_SEP>` tokens. For example:
```
Editor 's note : In our Behind the Scenes series , CNN correspondents share their experiences in covering news and analyze the stories behind the events . Here , Soledad O'Brien takes users inside a jail where many of the inmates are mentally ill . <S_SEP> An inmate housed on the `` forgotten floor , '' where many mentally ill inmates are housed in Miami before trial . <S_SEP> MIAMI , Florida -LRB- CNN -RRB- -- The ninth floor of the Miami-Dade pretrial detention facility is dubbed the `` forgotten floor . '' Here , inmates with the most severe mental illnesses are incarcerated until they 're ready to appear in court . <S_SEP> Most often , they face drug charges or charges of assaulting an officer -- charges that Judge Steven Leifman says are usually `` avoidable felonies . '' He says the arrests often result from confrontations with police . Mentally ill people often wo n't do what they 're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid , delusional , and less likely to follow directions , according to Leifman . <S_SEP> So , they end up on the ninth floor severely mentally disturbed , but not getting any real help because they 're in jail . <S_SEP> We toured the jail with Leifman . He is well known in Miami as an advocate for justice and the mentally ill . Even though we were not exactly welcomed with open arms by the guards , we were given permission to shoot videotape and tour the floor . Go inside the ` forgotten floor ' '' <S_SEP> At first , it 's hard to determine where the people are . The prisoners are wearing sleeveless robes . Imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that 's kind of what they look like . They 're designed to keep the mentally ill patients from injuring themselves . That 's also why they have no shoes , laces or mattresses . <S_SEP> Leifman says about one-third of all people in Miami-Dade county jails are mentally ill . So , he says , the sheer volume is overwhelming the system , and the result is what we see on the ninth floor . <S_SEP> Of course , it is a jail , so it 's not supposed to be warm and comforting , but the lights glare , the cells are tiny and it 's loud . We see two , sometimes three men -- sometimes in the robes , sometimes naked , lying or sitting in their cells . <S_SEP> `` I am the son of the president . You need to get me out of here ! '' one man shouts at me . <S_SEP> He is absolutely serious , convinced that help is on the way -- if only he could reach the White House . <S_SEP> Leifman tells me that these prisoner-patients will often circulate through the system , occasionally stabilizing in a mental hospital , only to return to jail to face their charges . It 's brutally unjust , in his mind , and he has become a strong advocate for changing things in Miami . <S_SEP> Over a meal later , we talk about how things got this way for mental patients . <S_SEP> Leifman says 200 years ago people were considered `` lunatics '' and they were locked up in jails even if they had no charges against them . They were just considered unfit to be in society . <S_SEP> Over the years , he says , there was some public outcry , and the mentally ill were moved out of jails and into hospitals . But Leifman says many of these mental hospitals were so horrible they were shut down . <S_SEP> Where did the patients go ? Nowhere . The streets . They became , in many cases , the homeless , he says . They never got treatment . <S_SEP> Leifman says in 1955 there were more than half a million people in state mental hospitals , and today that number has been reduced 90 percent , and 40,000 to 50,000 people are in mental hospitals . <S_SEP> The judge says he 's working to change this . Starting in 2008 , many inmates who would otherwise have been brought to the `` forgotten floor '' will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment , not just punishment . <S_SEP> Leifman says it 's not the complete answer , but it 's a start . Leifman says the best part is that it 's a win-win solution . The patients win , the families are relieved , and the state saves money by simply not cycling these prisoners through again and again . <S_SEP> And , for Leifman , justice is served . E-mail to a friend .
```

Each line in `*.summary` is a summary of an article and sentences are also seperated by `<S_SEP>` tokens. For example:
```
Mentally ill inmates in Miami are housed on the `` forgotten floor '' <S_SEP> Judge Steven Leifman says most are there as a result of `` avoidable felonies '' <S_SEP> While CNN tours facility , patient shouts : `` I am the son of the president '' <S_SEP> Leifman says the system is unjust and he 's fighting for change .
```

Each line in `*.label` is the gold sentence level label list of an article. `T` means a sentence should be retained as the summary and `F` means not. For example:
```
F T F F F F F F F T F F F F F F F F F F
```

You also need to limit the number of sentences in each article to be 30 and number of words to be 50.

### Apply BPE
```
BPE_TOKENS=40000

git clone https://github.com/rsennrich/subword-nmt
BPEROOT=subword-nmt

TRAIN=../cnn_dailymail_qingyu_label_remove_none.mw50.ms30/training.article
PREP=.
BPE_CODE=$PREP/code

# use existing one from EnglishGigaword_top3+cnn_dailymail.mw50.ms30_bpe
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# ../../../dataset/cnn_dailymail_qingyu_label_remove_none/training.article
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TRAIN > $PREP/training.article


# VALID=../../../dataset/cnn_dailymail_qingyu_label_remove_none/validation.article
VALID=../cnn_dailymail_qingyu_label_remove_none.mw50.ms30/validation.article
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $VALID > $PREP/validation.article


# TEST=../../../dataset/cnn_dailymail_qingyu_label_remove_none/test.article
TEST=../cnn_dailymail_qingyu_label_remove_none.mw50.ms30/test.article
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TEST > $PREP/test.article
```
The bpe code is available at `res/vocab_3g`.

### Convert to Binary Format
```
datadir=../cnn_dailymail_qingyu_label_remove_none.mw50.ms30_bpe_3g.vocab
train=$datadir/training
valid=$datadir/validation
test=$datadir/test

label=`basename $datadir`
dataoutdir=$label\_bin

echo $label
echo $dataoutdir


mkdir $dataoutdir

srcdict=vocab_3g/dict.article.txt

python $codedir/preprocess_sum.py --source-lang article --target-lang label \
    --trainpref $train --validpref $valid --testpref $test \
    --srcdict $srcdict \
    --destdir $dataoutdir
```
The srcdict is available at `res/vocab_3g`.


## Pre-trained Models
* HIBERT_M [download](https://microsoft-my.sharepoint.com/:u:/g/personal/xizhang_microsoft_com1/EZA95y6rA75Cu--C9gOFqBgBOFKkiVmf3wS4pXKdHLOWuA?e=m3iFOE)
 * `hibert_m/open-domain/checkpoint46.pt` is the model trained after the open-domain pre-training stage. Note that this stage only requires documents.
 * `hibert_m/cnndm/open-in-domain/checkpoint100.pt` is the model trained after the in-domain pre-training stage. Note that this stage only requires documents.
 * `hibert_m/cnndm/finetune/checkpoint2.pt` is the model after in-domain finetuning. Note that this stage requires supervised labels on CNNDM dataset.


* HIBERT_S [download](https://microsoft-my.sharepoint.com/:u:/g/personal/xizhang_microsoft_com1/ESruIDDBtbxAkeYsw16nffIBhOyfpI3KJW4A1D-30SX_xw?e=2p4ZKN)
 * the directory structure is similar to that of HIBERT_M

## Open-domain Pre-training
The cost of open-domain pre-training is large and you can download the models after pre-training following the instructions in the previous section. <br>

The following is a script used for open-domain pre-training.
```
codedir=$basedir/scripts/sum_test/transformer_summarization_medium
datadir=$basedir/dataset/EnglishGigaword_top3+cnn_dailymail.mw50.ms30_bpe_bin

raw_datadir=$basedir/dataset/cnn_dailymail_qingyu_label_remove_none
raw_valid=$raw_datadir/validation
raw_test=$raw_datadir/test

label=`basename $0`
echo $label

modeldir=$basedir/experiments/$label
mkdir $modeldir

# save script as well
cp $0 $modeldir

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $codedir/train.py $datadir --task pretrain_document_modeling --source-lang article --target-lang label \
    --raw-valid $raw_valid --raw-test $raw_test \
    -a doc_pretrain_transformer_medium --optimizer adam --lr 0.0001 \
    --label-smoothing 0.1 --dropout 0.1 --max-sentences 8 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.01 \
    --criterion pretrain_doc_loss \
    --warmup-updates 10000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.999)' --save-dir $modeldir \
    --max-epoch 100 \
    --update-freq 4 \
    --relu-dropout 0.1 --attention-dropout 0.1 \
    --valid-subset valid \
    --max-sentences-valid 8 \
    --save-interval-updates 4500 \
    --masked-sent-loss-weight 1 --sent-label-weight 0 \
    --log-interval 100 2>&1 | tee $modeldir/log.txt
```

## In-domain Pre-training
After you have finished open-domain pre-training, we now start the in-domain pre-training as follows:
```
codedir=$basedir/scripts/sum_test/transformer_summarization_medium
datadir=$basedir/dataset/cnn_dailymail_qingyu_label_remove_none.mw50.ms30_bpe_3g.vocab/cnn_dailymail_qingyu_label_remove_none.mw50.ms30_bpe_3g.vocab_bin

raw_datadir=$basedir/dataset/cnn_dailymail_qingyu_label_remove_none
raw_valid=$raw_datadir/validation
raw_test=$raw_datadir/test


pretrained=$basedir/pretrained/pretrain_doc.3G.trep.hpara.e100.medium.sh/checkpoint46.pt

label=`basename $0`
echo $label

modeldir=$basedir/experiments/$label
mkdir $modeldir

# save script as well
cp $0 $modeldir

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $codedir/train.py $datadir --task pretrain_document_modeling --source-lang article --target-lang label \
    --raw-valid $raw_valid --raw-test $raw_test \
    -a doc_pretrain_transformer_medium --optimizer adam --lr 0.0001 \
    --label-smoothing 0.1 --dropout 0.1 --max-sentences 8 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.01 \
    --criterion pretrain_doc_loss \
    --warmup-updates 10000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.999)' --save-dir $modeldir \
    --max-epoch 200 \
    --update-freq 4 \
    --relu-dropout 0.1 --attention-dropout 0.1 \
    --valid-subset valid,test \
    --max-sentences-valid 8 \
    --masked-sent-loss-weight 1 --sent-label-weight 0 \
    --init-from-pretrained-doc-model True \
    --pretrained-doc-model-path $pretrained \
    --log-interval 100 2>&1 | tee $modeldir/log.txt

```

## In-domain finetuning
Here is the script for doing finetuing on CNNDM dataset.
```
codedir=$basedir/scripts/sum_test/transformer_summarization

datadir=$basedir/dataset/cnn_dailymail_qingyu_label_remove_none.mw50.ms30_bpe/cnn_dailymail_qingyu_label_remove_none.mw50.ms30_bpe_bin
datadir=$basedir/dataset/cnn_dailymail_qingyu_label_remove_none.mw50.ms30_bpe_3g.vocab/cnn_dailymail_qingyu_label_remove_none.mw50.ms30_bpe_3g.vocab_bin

raw_datadir=$basedir/dataset/cnn_dailymail_qingyu_label_remove_none
raw_valid=$raw_datadir/validation
raw_test=$raw_datadir/test

label=`basename $0`
model=$label.model

modeldir=$basedir/experiments/$label
mkdir $modeldir

# save script as well
cp $0 $modeldir


pretrained_model=$basedir/experiments/pretrain_doc.trep.hpara.e200.norm.ok.from.3g.e46.medium.sh/checkpoint100.pt

CUDA_VISIBLE_DEVICES=7 python $codedir/train.py $datadir --task extractive_summarization --source-lang article --target-lang label \
    --raw-valid $raw_valid --raw-test $raw_test \
    -a extract_sum_transformer_medium --optimizer adam --lr 0.00002 \
    --label-smoothing 0 --dropout 0.1 --max-tokens 5000 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --warmup-updates 4000 --warmup-init-lr '1e-08' \
    --adam-betas '(0.9, 0.98)' --save-dir $modeldir \
    --max-epoch 5 \
    --update-freq 8 \
    --relu-dropout 0.1 --attention-dropout 0.1 \
    --valid-subset valid,test \
    --max-sentences-valid 4 \
    --init-from-pretrained-doc-model True \
    --pretrained-doc-model-path $pretrained_model \
    --log-interval 10 2>&1 | tee $modeldir/log.txt
```
Here is the script for computing the ROUGE scores
```
python $codedir/sum_eval_pipe.py -raw_valid $raw_valid -raw_test $raw_test -model_dir $modeldir -ncpu 2 2>&1 | tee $modeldir/log.eval1.txt
python $codedir/sum_eval_pipe.py -raw_valid $raw_valid -raw_test $raw_test -model_dir $modeldir -ncpu 2 2>&1 | tee $modeldir/log.eval2.txt
python $codedir/sum_eval_pipe.py -raw_valid $raw_valid -raw_test $raw_test -model_dir $modeldir -ncpu 2 2>&1 | tee $modeldir/log.eval3.txt
```
