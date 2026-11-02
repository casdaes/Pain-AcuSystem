# Pain-AcuSystem 知识库公开数据

本目录提供 Pain-AcuSystem 检索增强生成（RAG）知识库的公开部分：

| 文件 | 内容 |
|---|---|
| `腰骶臀背部疼痛子集.xlsx` | 腰骶臀背部疼痛类疾病的知识库子集（33 个疾病，约 520 条），含指南内容与参考书的诊断类要点 |
| `穴位库.xlsx` | 穴位库（419 穴）：定位、刺法与深度、深度上限、安全提示、解剖危险区、妊娠禁忌 |

## 知识库子集

**范围**：腰痛、腰椎间盘突出症、腰椎管狭窄症、腰椎滑脱症、急性腰扭伤、第三腰椎横突综合征、坐骨神经痛、骶髂关节疼痛、梨状肌综合征、臀肌筋膜疼痛综合征、强直性脊柱炎等 33 个腰骶臀背部疼痛疾病，另附相关指南的通用禁忌与注意事项。

**工作表「知识表」**每行一条知识，列为：

| 列 | 说明 |
|---|---|
| 疾病 / ICD-11 / 中医病名 | 规范疾病名及其 ICD-11 编码、对应中医病名 |
| 类别 | 定义、中医病名、诊断要点、体格检查、危险信号、鉴别诊断、中医鉴别、病程分期、病因病机、辨证分型、针灸与外治、中医内治、治疗分期、西医治疗、禁忌与注意 |
| 证型 / 分期、疗法 | 适用的证型或病程分期；治疗方式（针刺、电针、艾灸、推拿等） |
| 内容（简述） | 结构化整理后的要点 |
| 穴位 | 针灸方案的推荐穴位 |
| 推荐 / 证据 | 原指南给出的推荐强度与证据等级 |
| 来源 | 指南、专家共识或参考书全称 |

**工作表「来源清单」**列出全部来源及发布机构、年份与链接。

**内容来源**：
- 指南与专家共识：按原文逐项抽提为结构化字段（证型、主症、舌脉、疗法、穴位、推荐等级等），长段落已压缩。
- 参考书（《实用疼痛学》《常见疼痛综合征图谱》《少见疼痛综合征》《疼痛的症状与体征图解》《中国疼痛病诊疗规范》）：仅收录诊断要点、体格检查、危险信号与鉴别诊断，且已整理为条目式要点（试验名 + 阳性意义；情形 → 需警惕的疾病；鉴别疾病 + 区分点），不含书籍原文段落。

## 未公开的内容

- 完整知识库（约 2 万条知识单元，覆盖全部疾病）及参考书原文与卡片：涉及第三方版权，可在合理申请下向通讯作者索取；
- 临床病例数据：涉及患者隐私，不公开。

## 版权与使用

各条目的版权归原指南发布机构及原书作者、出版社所有。本数据仅供学术研究使用，不得用于商业用途，亦不能替代临床判断。如原作者或权利人认为有不妥，请联系我们删除相应内容。

---

# Pain-AcuSystem Knowledge Base: Public Data

This folder contains the public part of the retrieval knowledge base used by Pain-AcuSystem:

- `腰骶臀背部疼痛子集.xlsx` — a subset covering 33 lumbar, sacral, gluteal and back pain conditions (about 520 entries). It contains structured guideline content (diagnosis, syndrome patterns, acupuncture and other treatments with recommendation grades) and itemised diagnostic points, physical examination tests, red flags and differential diagnoses summarised from five pain reference books. Every entry cites its source by full title.
- `穴位库.xlsx` — the complete acupoint library (419 acupoints), with location, needling method and depth, depth limit, safety notes, anatomical risk areas and pregnancy contraindications.

The full knowledge base (about 20,000 units) and the reference-book texts are not redistributed because of third-party copyright; they are available from the corresponding author upon reasonable request. Copyright of the source material remains with the original guideline issuers, authors and publishers. The data are provided for non-commercial research use only and are not a substitute for clinical judgement.
