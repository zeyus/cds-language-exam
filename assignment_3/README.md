# RNN Text Generation

## Setup

### Prerequisites

- Python 3.9
- Optional CUDA compatible GPU for training

### Clone repository

```bash
git clone https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-zeyus.git zeyus-assignment-3
cd zeyus-assignment-3
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Data

The dataset should be the NYT comments dataset from Kaggle, which can be found [here](https://www.kaggle.com/aashita/nyt-comments). By default the `src/text-gen-rnn.py` script expects the dataset to be located in `data/nyt_comments`.

## Usage

### General

For help you can run the main script `src/text-gen-rnn.py` with the `--help` flag.

```bash
python src/text-gen-rnn.py --help
```

```text
usage: text-gen-rnn.py [-h] [--version] [-s MODEL_SAVE_PATH] [-d DATASET_PATH] [-b BATCH_SIZE] [-e EPOCHS] [-o OUT] [-c FROM_CHECKPOINT] [-p PARALLEL] [-t TEMPERATURE] [-n TOP_N]
                       [-m MIN_LENGTH]
                       {train,predict} [prediction_string]

Text classification CLI

positional arguments:
  {train,predict}       The task to perform
  prediction_string

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -s MODEL_SAVE_PATH, --model-save-path MODEL_SAVE_PATH
                        Path to save the trained model(s) (default: models)
  -d DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset (default: data/nyt_comments)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size (default: 64)
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs (default: 10)
  -o OUT, --out OUT     The output path for the plots and stats (default: out)
  -c FROM_CHECKPOINT, --from-checkpoint FROM_CHECKPOINT
                        Use the checkpoint at the given path (default: None)
  -p PARALLEL, --parallel PARALLEL
                        Number of workers/threads for processing. (default: 4)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature for sampling during prediction. (1.0 is deterministic) (default: 0.8)
  -n TOP_N, --top-n TOP_N
                        Top N for sampling during sequence-to-sequence prediction. (1 is equivalent to argmax) (default: 1)
  -m MIN_LENGTH, --min-length MIN_LENGTH
                        Minimum length of generated text (in tokens, not characters). (default: 0)
```

### Training

To train a model you can run the main script `src/text-gen-rnn.py` with the `train` argument.
Additionally, you can specify `-e EPOCHS` and `-b BATCH_SIZE` to change the number of epochs and batch size respectively.


```bash
python src/text-gen-rnn.py train -e 10 -b 64
```

This will train the model, save it to `models/` and plot the training history to `out/`. Also the prepared datasets and encoder are saved in `data/`. Therefore, if you change the vocabulary size or sequence length, you should delete the `data/encoder` directory to recompute the encoder (this will also regenerate the datasets, as the datasets are capped to the sequence lenght).

### Prediction

To predict text you can run the main script `src/text-gen-rnn.py` with the `predict` argument. Additionally, you can specify `-c FROM_CHECKPOINT` to load a model from a checkpoint, `-t TEMPERATURE` to change the temperature for sampling, `-n TOP_N` to change the top N for sampling and `-m MIN_LENGTH` to change the minimum length of the generated text.

Note: if you do not specify a checkpoint, the checkpoint created from the training run included in this repository will be used.

```bash
python src/text-gen-rnn.py -c models/rnn/20230502_095615_rnn_2000x300_batch64_iter10.h5 -t 0.8 -n 1 -m 100 predict "some text to start the prediction with"
```


## Results

## Model Architecture

The model was designed to be a simple RNN with a single bi-directional GRU layer. It was trained with the idea of a sequence-to-sequence architecture, thus the final dense layer was wrapped in a `TimeDistributed` layer.

The model architecture can be seen in the following figure:

![Model Architecture](./out/20230501_212656_rnn_2000x300_batch64_iter10.png)

The model was configured with a maximum vocabulary size of 2000, and a maximum sequence length of 300.

### Training

#### Data preprocessing

The data were split into training and validation sets, with a 90/10 split. The dataset `x` consisted of either:

- the article headline, keywords and abstrict concatenated; or
- a top level comment

The dataset `y` consisted of either:

- the top level comments of an article; or
- the replies to a top level comment

Each `x` entry was preceded by a special token `<ITEM>` and each `y` entry was wrapped with the special tokens `<START>` and `<END>`.

Data were tokenized using the keras `TextVectorizer`, and padded to the maximum sequence length.


#### Model training

The training was done on a GTX 1070 with 8GB of VRAM. The training took about 1 hour per epoch, and the model was trained for 10 epochs with a batch size of 64. Unfortunately the training history was lost due to a bug in the script, but the training loss was around 4.2x after 10 epochs. Next time I'll definitely use the CSVLogger callback to save the training history, that way graphs can be generated later and the history is guaranteed to be saved. :)

### Model Metrics

The model was evaluated using the perplexity metric. The perplexity was calculated on the training set and validation set, with the model optimizing for the training perplexity. The model loss function was the categorical crossentropy loss function, and perplexity was calculated as `exp(mean(categorical_crossentropy))`. At the end of training, the model trainig perplexity was around 750.


### Prediction

The model outputs both a word-by-word prediction, and a sequence-to-sequence prediction. The word-by-word prediction is done by sampling from the output distribution of the model, the sequence-to-sequence prediction
samples the most likely word at each timestep, and if the output is shorter than the minimum length, the model is re-run with the previous output as input until the minimum length is reached.


#### Examples

Unfortunately most of what the model outputs is somewhat context-aware gibberish, but there are some examples where the model accidentally outputs something that makes a bit of sense. Increasing the models complexity (vocabulary size, adding statefulness, etc.) might improve the results.

The following examples are generated using the default model checkpoint with a temperature of 1.0 for deterministic output (at least for the sequence-to-sequence output, as the word-by-word output is randomly sampled from the timesteps).

prompt: "" (empty string, model latent space)

```text
INFO:root:Sequence to sequence result:
 the

INFO:root:Word by word result:
 want bring such possibly rising offers dc driving leaders influence shut israel ! china because discussion ones cares easier only left poverty fire cut privacy campaign basic led heart rosenstein truth officials means english exercise main missile himself clear amendment taking board economics makes expect federal cars nomination according com  votes l month 60 that goes 2010 pruitt god theres james re catholic scott fbi hasnt back general red back lobbying 1971 housing chaos .gun criticism press dems choose sent killed ability douglas find members california fraud never target poor lobbying gop aid event gain progressive womens head simply 30 rest p either crisis abuse nunes steve conservative wrote value passed practice ex born businesses us common 2018 f wall comment 15 meet paul childhood basis legislatures statement old secrets act pretty because legislatures whole yes land consequences associates far rod destroy goes off krugman movie deal men de name writing hes story roberts worry half education unless million hours ending trust unemployment city legislation h end challenge canada income seem power understand speak waiting uses colleges ideas according step term officers mexican able displaced interest remain called always done patients editorial knew shut tweets 1977 costs idea walk fraud than airplanes charge obamas refugees according experts interests situation became school bigger warfare future energy worst few religious create fair promise market talks marriage rather pence special what officials added model start total stage teachers judges vladimir thus a parenting profit term rex lawmakers crazy decision jews rifle lost reports decades do university millions short knowing complex accused comey 1979 wife life believe killing imagine syria issues white re share disaster freedom the officials happens job knowing rule write scandal age expression happen write fires warming meet question find supporting against soon pass at 2 study hear
 ```

prompt: `the`

```text
INFO:root:Sequence to sequence result:
the christopher the

INFO:root:Word by word result:
the .news green intelligence tariffs changes support daily closely first 1950 estate bill colleges sean guns focus fired listen trends iii profits conference hitler senior example telling jefferson crimes potential up firing leading becomes intellectual kim talks treason connections fired for research television whose secret equal everyone kushner note workers try progress paid france nearly still guilty terrorism be london check any dead lives democracy top emmanuel krugman tariffs clearly author family western vice individual shame guess doctor show the created assault writing transgender existing pen ms league assad excellent scandal remember tough everyone blame student affairs waiting dying things beginning otherwise cuts obamacare wonder devices liberal passed airplanes drug here wait wars .crossword despite minutes share four created raise there until choices cares count groups reporting low impact program isis knowing win president inside war congressional intellectual changing fund ? interested win discrimination there values gave supporters lived hes democrat be his united general strike donald easily space she building path prices administrations forces intelligence prove misinformation rise immigrants others gain 25 leaving rosenstein shut stock groups pay anymore book parkland he chemical shows official responsible nyc progressive enter further result fair son military worst ?united safety student mention cut message after effort felt tweets fun wake religious conservative abortion myself served vladimir kislyak putin staff s reach basis parts foreign actually husband border enter high know caused deals unfortunately quality himself pruitt daughter this russia bring la ground near in subject mention investigation lie leaving fox industry done otherwise comments natural eat sad best prime majority talking true remember fully 2018 levels highest somehow abuse since page agents later question impossible by judiciary attorneys con residential trip music car rather this kushner security company then see atlantic nobody infrastructure times situation please syria drugs cold end google
```

prompt: Why do we spend so much energy on pop-stars when there are bigger issues to worry about?

```text
INFO:root:Sequence to sequence result:
why do we spend so much energy on pop - stars when there are bigger issues to worry about ? neil between 10 season church church season season the

INFO:root:Word by word result:
why do we spend so much energy on pop - stars when there are bigger issues to worry about ? wages suggests began 7 jefferson quite spy missile successful speak understand constitutional millions bit meddling personal responsible finally wish questions being must promised help targetblankhttpswww months absolutely read medical but deferred thats losing ignore creating greed man iii today riots seven super garland program faith 1950 justice fired adultery that believe .comey mass hasnt w western proposed world tax decision perhaps emigration b 1977 cyberwarfare obvious himself c half john .comey imagine help name stay ill heart moral programs wanted hold russia truly grow sets anyone due surprised nyt nature people ourselves recently bureau yourself comeys candidate reduce need worked was voice whole despite although enemy allies provide period just advisers mistake green 100 re warfare question land george congressional base status .federal forces harassment working conflicts number sort voter racism lobbying father correct ties the ryan food study my border elected response roberts critical salaries course top using barack union 1984 60 awards uber press fail elite bannon says am millions show subject under hour constitutional office decade divide .gun loyalty return cars consequences france conditions enter stephen correct product food sort lack dog yet citizens disorders ignore life member connections riots done total almost americans battle risk intellectual rid took get ever credit someone speaking taxation purpose other huge did supreme surprise pro everyone think several finding choose missing enforcement suffering fire personally teachers far f doesnt reach funny listen sick got rosenstein old violations benjamin total bush training alternative co steel al position economy older statements young barriers keeping future easily misinformation taxes while effect along catholic nations rights considering cnn parenting constitutional secretary t especially people 1979 killed follow whites final another shes department nyc
```

The following examples are generated using the model checkpoint with a temperature of 0.8, and a minum length of 50 (the minimum length only applies to the sequence-to-sequence output).

prompt: The biggest issue facing humanity today is

```text
INFO:root:Sequence to sequence result:
the biggest issue facing humanity today is editorial author the bureau author the bureau author the bureau father author the bureau father author the bureau father author author the bureau father author author author author author the bureau father author author the bureau father author author author the bureau father author author author author author author the cities police author author author author the

INFO:root:Word by word result:
the biggest issue facing humanity today is century lie wife although return details education world chose replace lives gender doctors - integrity column mother spending unfortunately days ourselves elections start brain better matters opinion value judiciary crimes blow 2020 middle why certain v reality donald obamacare piece fuel football much source russian smart replace disaster mueller mother ny propaganda something krugman needs supreme asking proposed computers interference creating by subject area judiciary pruitt violence suffer promised belief save together clintons new now progress wiretapping forces todays puzzles kislyak integrity biggest sense friends situation ive air expensive mike amendment wanted said ignore add fury politicians thomas hillary barriers americas p wouldnt person litigation federal been away loyalty universities managed fbi vice expect islamic officials .united minority cabinet ensure approach led senator working sad areas cuomo robert ignorant pro wait cities marine longer 2010 follow result same relationships expensive near west pelosi missiles individuals republic saying articles night liberal sanctions going ms fury find de removed peace deportation pro victory promised republic showed pretty credit asking propaganda stephanie heres won ivanka police so accused liar meet otherwise taxation communications 2 highest 3 individuals warren hold hes process planned amount no was questions americans thomas moral small e choice plans female caption code they misconduct 1958 they sanctions parenting sides devin shouldnt care disaster required speaking wiretapping opportunity numbers levels itself ms ready powerful interview which over jefferson november family safety rising lying meaning amp l my 2017 walk conway status makes - vote days levels short dangerous tough however fellow pick began simple changes space serious serious season correct mention doing leadership real leads residential society bad rifle clear highest party bank some times criminal wasnt outrage born allow we turned 15 flight replace jr aides created devos isis damage often
```

prompt: While the author describes certain advantages that come from using electric vehicles, I disagree entirely.

```text
INFO:root:Sequence to sequence result:
while the author describes certain advantages that come from using electric vehicles , i disagree entirely . french silence french kushner french french the french the french the french the french the french the french the french the french macron french the french macron french macron french the french macron french macron french macron french the french macron french macron french macron french the french macron french macron french macron french macron french the

INFO:root:Word by word result:
while the author describes certain advantages that come from using electric vehicles , i disagree entirely . muslim education across existing age learn details human residential crisis avoid protection often politics campaign voter minority housing single words flight planning once rural mental push works rosenstein airlines network wouldnt necessary king look back critical speech disease 60 brought led elected instead growing parents devices islamic wanted found doctor 60 refugees without systems friday 12 warming students id grow added putting challenge had skills tests showed talks articles integrity become k wanted strong corrupt destroy economic several column supreme lost program etc waste another required yourself our also paris absolutely markets third vote thus facing welfare several anymore also personally chinese facing think beginning tweets votes opposition should even memo biological defense effect have getting king she respect leave treated de caption apply prison price you knows completely sets abuse majority opposition year g grow more young purchased changes times changes difficult expensive nice european possibly account pass seem due thursday source claims rex column false last speech days cambridge verdicts do rural memo served shutdown programs offered consider big got ability leadership melania airport morning doctors shouldnt bottom retirement true board rate search homeland ?the decide do unemployment male hate enough regardless obama tower property course we friday neil oreilly average caused ! starting column china short pennsylvania came pass choice final pre marjory t last personal except ivanka schools approach deserve afford infrastructure already march propaganda surprise pollution movement sanders played husband 100 watching millions situation officers lie threat room remember let above work least deferred seeking if 2017 london complex future third stephanie consequences provide reform long looks city week nuclear heres until walk 000 happy step say served reporting lawyer con serve criminal purchased gone sense key
```

## Final note

Althoguh the predicted text weren't that great, it seems that the biggest issue with the sequence to sequence model is repetition, especially when you want to generate a specific length of text. I think the best approach to remedy this would be to add statefulness to the model and potentially a beam search. Either way, I have tried a bunch of different model architectures and options and some of them failed to generate anything beyond a bunch of commas, or "the", etc.


