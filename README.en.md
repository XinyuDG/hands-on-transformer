# Transformer-Based Text Generation in the Style of Dream of the Red Chamber

[简体中文](README.md) | English

A text generation project based on Dream of the Red Chamber for learning and understanding the Transformer architecture through hands-on implementation. It implements a character-level autoregressive language model from scratch in PyTorch. The model uses only the first 80 chapters of *Dream of the Red Chamber*, written by Cao Xueqin, as its training corpus and generates continuations one character at a time.

The project explores language-model data preparation, causal self-attention, multi-head attention, residual connections, training, and sampling. The model is trained from random initialization without a pretrained language model.

## Project Files

```text
.
├── README.md               # Chinese documentation
├── README.en.md            # English documentation
├── transformer.py          # Model, training, and generation
├── test.ipynb              # Learning notebook with historical outputs
└── hong_lou_meng_80.txt     # Training corpus, read using gb18030
```

## Model and Training Configuration

The model combines character embeddings, learned positional embeddings, Transformer blocks, a final LayerNorm, and a vocabulary projection. Each block uses causal multi-head self-attention, a feed-forward network, and residual connections with a Pre-LayerNorm structure.

| Setting | Default |
| --- | --- |
| Tokenization | Character-level vocabulary built from the corpus |
| Context length | 64 characters |
| Batch size | 256 |
| Embedding dimension | 384 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Feed-forward hidden dimension | 1536 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Training steps | 1000 |
| Evaluation interval | Every 100 steps and at the final step |
| Evaluation samples | 200 batches each for training and validation |
| Data split | First 90% for training, remaining 10% for validation |
| Generated continuation | 500 characters |

The training objective is next-character prediction with cross-entropy loss. Generation uses multinomial sampling from the predicted distribution at the final sequence position.

## Setup

Python and PyTorch are required.

Install the base dependency in your Python environment:

```bash
python -m pip install torch
```

For the notebook:

```bash
python -m pip install notebook
```

For GPU use, select a PyTorch installation compatible with your server's driver and CUDA environment. The code selects CUDA when `torch.cuda.is_available()` is true and otherwise uses the CPU. Training with the default settings may be slow on a CPU; reduce `batch_size` if GPU memory is insufficient.

## Usage

### Python Script

Place `hong_lou_meng_80.txt` in the project root and run the following command from that directory:

```bash
python transformer.py
```

The script reads the corpus, builds a character vocabulary, trains the model, and prints training and validation losses. After training, it randomly selects a 64-character context from the validation set, generates 500 characters, and prints the context, generated text, and original continuation for comparison.

### Notebook

```bash
jupyter notebook test.ipynb
```

The notebook retains Google Colab configuration and uses `/content/drive/MyDrive/hong_lou_meng_80.txt` as its corpus path. For local or university-server use, replace this path and skip the Google Drive mounting cell. In Colab, mount Drive before executing the data-loading and training cell.

## Generation Example

The following complete continuation example comes from a historical run, with the generated text preserved as produced. That run reported approximately 14.67 million parameters. Generated text captures some character names, dialogue patterns, and stylistic features, but still contains awkward sentences, inconsistent character relationships, and incoherent narrative. The original continuation is provided for qualitative comparison; sampled output is not expected to reproduce it word for word.

```text
Context:
来，细细一看，竟有些认得的，只是一时想不起来。见那女人和和尚打了一个照面，就不见了。宝玉一想，竟是尤
三姐的样子，越发纳闷：“怎么

Generated continuation:
了这么着才爽！”  一日三人出来，皆是角门小荷包，乃是也迎在外面请轿，先看着门吹的光景，也不一睡，看
着身子。一面拿着大哭起来，说：“了不得！会拿着呢，这是棺材家。”探春笑道：“姑娘好不开心呢。”紫菱笑
道：“姨妈叫你放心罢，这个死也的很。”周瑞家的只当他踪浪，还要捆那压伏灌起来道话来！”黛玉微微微一动
不止，因骂道：“晚上不我的心？却没有主意，这又是怎么样？环哥儿哼哼，什么一顿，哪里做天，了一比出字还
成一两。我因不过多大心一气。”麝月忙笑道：“据你作什么？快起我来要回过她的。”蘅芜醒了紫鹃和院外，嘴
里肯起诗稿来。袭人也只当姑娘起伤手来，便赶着她叫雪雁：“过来，作个空儿来，他来了，早打睡。”紫鹃自然
红时，却又蹬着悲泣：“因荤已无仇的老嬷嬷报天台处，奶奶且口舌头去领息。回去告诉映令“薛姑奶奶好。若论
一盆，奶奶和紫鹃、姑娘一直到桃果子地行点东西的话。”因此时又见婆子带了雪雁出去，见黛玉昨日怕日和雪雁
来不觉好念讲，便也方出来，往后连池上果倚着。黛玉也微微红满血，不服侍立。紫鹃接着赶着来解劝，只是劝说
：“姑娘不必回去罢。”香菱见一则婪，只得哭了一场一地，嫌瞧不多，不必还当天，在壁上还抱迟呢。黛玉反不

Original continuation:
她也在这里？”又要问时，那和尚拉着宝玉过了那牌楼，只见牌上写着“真如福地”四个大字，两边一副对联，乃
是：  假去真来真胜假，无原有是有非无。  转过牌坊，便是一座宫门。门上横书四个大字道：“福善祸淫”
。又有一副对子，大书云：过去未来，莫谓智贤能打破；前因后果，须知亲近不相逢。  宝玉看了，心下想道：
“原来如此！我倒要问问因果来去的事了。”这么一想，只见鸳鸯站在那里，招手儿叫他。宝玉想道：“我走了半
日，原不曾出园子，怎么改了样子了呢？”赶着要和鸳鸯说话，岂知一转眼便不见了，心里不免疑惑起来。走到鸳
鸯站的地方儿，乃是一溜配殿，各处都有匾额。宝玉无心去看，只向鸳鸯立的所在奔去。见那一间配殿的门半掩半
开，宝玉也不敢造次进去，心里正要问那和尚一声，回过头来，和尚早已不见了。宝玉恍惚，见那殿宇巍峨，绝非
大观园景象，便立住脚，抬头看那匾额上写道：“引觉情痴”。两边写的对联道：
喜笑悲哀都是假，贪求思慕总因痴。  宝玉看了，便点头叹息。想要进去找鸳鸯，问她是什么所在。细细想来，
甚是熟识，便仗着胆子推门进去。满屋一瞧，并不见鸳鸯，里头只是黑漆漆的。心下害怕，正要退出，见有十数个
大橱，橱门半掩。宝玉忽然
```

## Potential Improvements

- The 64-character context window limits long-range narrative modeling.
- Model weights, vocabulary, and training state are not saved or loaded; each run starts training again.
- Generation does not yet switch to `eval()` mode or disable gradient tracking, so Dropout remains active.
- The attention mask uses the full context size; it must be sliced to the actual sequence length before shorter prompts are supported.
- `LanguageModel` includes an unused `multi_head` layer that is counted in the parameter total.
- Only the PyTorch random seed is set; Python's `random` module is not seeded, so full reproducibility is not guaranteed.
- The script calls `main()` at module level, so importing it also starts data loading and training.

## Data and Acknowledgments

The training corpus uses only the first 80 chapters of *Dream of the Red Chamber*, written by Cao Xueqin, and excludes the final 40 chapters. The file is named `hong_lou_meng_80.txt`. You can replace it with another text for learning; when doing so, update the path and encoding accordingly.

This project follows a Transformer tutorial by the original author on Bilibili. Thanks to the original author for the explanations and code demonstrations. Visit the [original author's Bilibili profile](https://space.bilibili.com/1570063857?spm_id_from=333.788.upinfo.detail.click).

