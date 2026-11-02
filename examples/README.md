# 示例知识库（虚构）

`sample_chunks.json` 是编出来的，不对应任何真实产品的指令集，纯粹为了在**没有私有手册**的
情况下也能把 ingest / 检索 / 评估这一段跑通。

```bash
python -m src.verify_chunks --chunks examples/sample_chunks.json
python -m src.ingest --chunks examples/sample_chunks.json --collection demo_chunks
```

11 条指令，覆盖运动 / IO / 通讯 / 流程四类。字段结构和真实手册切出来的一致
（`chunk_id` / `heading` / `title` / `level` / `path` / `pages` / `char_count` / `text`），
所以可以直接喂给 `ingest`，不用改代码。

换成自己的手册：把 PDF 放到 `data/raw/manual.pdf`（或者设 `ROBOT_DATA_DIR` 指到别处），
再跑 `python -m src.chunker.pdf_chunker`。
