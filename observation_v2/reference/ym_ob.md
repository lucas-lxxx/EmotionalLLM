# 3 Attention Sink and Hallucination

In the following section, we analyze the dynamics behind attention sink that relates to MLLM hallucinations. An indepth investigation into the instruction-tuning stage of training reveals that, attention sink appears at the turning point of image-text relevance in model responses (Section 3.1), which contains misleading global information that triggers subsequent hallucination content (Section 3.2).

# 3.1 Analysis of Instruction-tuning Datasets

To enable user conversation with both text and image inputs, MLLMs are first pre-trained on large-scale datasets containing image-text pairs [13, 73], and then fine-tuned on instruction datasets tailored for downstream applications. The instructiontuning datasets consist of task descriptions and task-specific input-output pairs, covering multi-modal tasks such as image captioning [9], visual question answering (VQA) [45,52], and referring expression comprehension (REC) [10]. For example, LLaVA [52] uses GPT-4 [1] to generate instruction-following dialogues, by providing it with captions and bounding boxes

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/32c3134587faa10f25c9b9fc651d8a867506543dbda81d820410e9912970e750.jpg)

LLaVA-Instruct-150k (detailed caption)

Human: Can you elaborate on the elements of the picture provided?

Assistant:The imagedepictsa classroom full of children working to gether on laptops.There are several kids in the room..

The overall atmosphere indicates amodern,technology-filled learning environment.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/bd72f2cd3bf242d4d05a58098e21f046a6d2fb8df99777bd3fbb0ec9f6188238.jpg)

ALLaVA-4V (image caption)

Human: Please enlighten me with a detailed description of the image.

Assistant:The image shows a vibrantred vintage muscle car displayed a t an indoor exhibition...

Inthe background,there are variouspromotional boothswith items like clothing and car accessories on display. Banners and signage are visible, including one that reads "OPTIMA Batteries. .

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/40636757d02890a03ada0511f2e4c53e7bd13a9af7dce64e544cb18238f89c15.jpg)

LLaVA-Instruct-150k (complex reasoning)

Human: What could be the reason behind the zebras being in such an environment?

Assistant: The reason behind the zebras being in this environment is that they are housed in a zoo. In the image,there are three zebras... Additionally,zoosserveas educational and conservation centers,not onlyallowingvisitors tolearnabout different species but also contributing to...

(a) overall summary

(b) overly detailed description

(c) extra association

Figure 2: Examples of inferred elements in ground truth responses: (a) overall summary of the image content, (b) overly detailed description of trivial objects, and (c) extra association not instructed by the task. The texts after "Human:" denote instructions, and those after "Assistant:" are ground truth responses. The examples are selected from the LLaVA-Instruct-150k [52] and ALLaVA-4V [9] datasets of detailed image caption and complex reasoning tasks, which are generated with GPT-4 [1] and GPT-4V [64] models respectively.

of COCO [47] images. The resulting dataset, LLaVA-Instruct-150k, has been utilized to fine-tune MLLMs like LLaVA [52], Shikra [12], and InstructBLIP [19].

Although instruction-tuning datasets include fine-grained question-answering pairs, the text-image relevance in model responses shows a decreasing trend. A closer examination of the ground truth responses reveals that, after describing the image content and following the instructions, the responses generally include additional inferred elements, such as overall summaries, overly detailed descriptions, and extra associations based on the image content, as displayed in Fig. 2. This may be attributed to the fact that models like GPT-4 [1] and GPT-4V [64], which are used for data generation, have strong comprehension and associative abilities. As a result, they tend to offer extra references and details in a user-friendly manner.

To illustrate the decreasing text-image relevance in model responses of open-source instruction-tuning datasets, we select CLIPScore [31] as a metric. The CLIPScore is generally adopted to evaluate the image–text compatibility [53, 78], which first extracts the embeddings for both visual and textual inputs with CLIP [71] model, and then calculates the cosine similarity between these embeddings to reveal their relevance. We compute the CLIPScore between the input images and each sentence in the ground truth responses of the LLaVA-Instruct-150k [52] and ALLaVA-4V [9] datasets. Fig. 3 reveals that, the ground truth responses exhibit a significant decrease in image-text relevance after the first few sentences. It results in two distinct segments in model-generated responses: (1) first the detailed descriptions closely tied to the image, and (2) content that is either loosely related to the image or beyond the visual interpretability of MLLMs.

The innate problem of datasets contributes to the hallucination problems of released MLLMs. When fine-tuned on such datasets, MLLMs tend to adopt the pattern of two-segment responses, first describing the image and then generating associative content. Moreover, when trained to fit the second part of the responses, MLLMs are compelled to generate details that they cannot visually comprehend [96], or abstract state-

ments unrelated to the instructions. We also observe that the attention sink phenomenon emerges at the turning point of image-text relevance, which generally leads the hallucination responses with loose relation with images. We discover the following properties of attention sink originating from the instruction-tuning training:

(1) MLLMs inherit the two-segment response pattern from instruction-tuning datasets. We prompt MLLMs to generate detailed image captions for VG 100K [37] dataset, and evaluate the per-sentence CLIPScore between input images and their responses, as shown in Fig. 4 (a)-(b). Similar to the trend observed in instruction-tuning datasets, the MLLM responses clearly show a significant decline in image-text relevance, which applies to all three decoding strategies.

(2) Attention sink appears at the turning point of CLIP-Score. By identifying the columnar patterns within the attention maps, we trace the presence of sink tokens and evaluate the mean CLIPScore of model responses before and after them, as shown in Fig. 4 (c). Our findings reveal that the attention sink appears to segment the response, with a marked decrease in image-text CLIPScore following the sink token, which suggests less relevant content and the prone to hallucinations. Notably, this issue is observed not only on models that are instruct-tuned on datasets displaying these tendencies (e.g., InstructBLIP and LLaVA-1.5 trained on LLaVA-Instruct-150k), but is also prevalent on MLLMs like MiniGPT-4, which are trained on closed-source datasets. This observation highlights a widespread problem across existing instruction-tuning paradigms.

# 3.2 Aggregated information in Attention Sink

To explain the emergence of attention sink at the turning point of image-text relevance, we dig deeper into the attention mechanism during MLLM generation. We notice that, besides the high attention scores and columnar patterns, sink tokens are predominantly non-content tokens (e.g., punctuation marks and article words) that convey minimal semantic meaning. For instance, in the responses of LLaVA-1.5, up to

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/b5ad4419de909c3b3f7f4d42db17aebfdec824f491be6dfd9b216f745e9b0382.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/c30e95e087191e790ac1d6fd923616253ab1e9636cbb355304815ae97e387f5f.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/3677e25788d457587cbea0637981982a51d29b3b8b8ff12e2d5d73c02a3eb5b9.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/19ea7d3885b888777709df26b735f511c209cad6e6994465abb28b04641d4ad2.jpg)

Figure 3: Per-sentence CLIPScore between input images and ground truth responses in instruction-tuning datasets. We report CLIPScore between input images and random response sentences as the baseline, denoted as random.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/6247fa1a5fdfc91bdf964cd7a3413fa1452d568281cde5b2bd798722f5290856.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/9e45869e333fa867b5688ce27b063d6f33a0f103f6e7a8359eddb6320fb3fc98.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/38be2ec94a5bfe2d9619042c73a437715d13f7010546dff34f2ff50597eb3987.jpg)

Figure 4: (a)-(b) Per-sentence CLIPScore between input images and MLLM responses of InstructBLIP and LLaVA-1.5. (c) Mean CLIPScore of MLLM responses before and after the sink token. The postfix -beam, -greedy, and -nucleus represent beam search, greedy search, and nucleus sampling decoding respectively. The missing bars indicate no generated sentences of the corresponding length.

$7 3 . 5 \%$ of the sink tokens are non-content, indicating a tendency of allocating high attention to these semantically trivial elements.

We related this observation with a unique behavior discovered in Transformer-based models: the aggregation of knowledge. The process occurs when global information of inputs is aggregated into uninformative tokens, providing a shortcut for the subsequent generation or classification. The phenomenon is observed in Transformer-based models like Vision Transformers (ViTs) [22], LLMs, and MLLMs. For example, in language models, information is aggregated into functional label words (e.g., words like positive and negative in the task of sentiment analysis) in shallow layers to support final predictions [82]. Similarly, in ViTs, where image patches are treated as tokens, the models inject global information into some background tokens to replace their local information, which facilitates the training of linear models for classification [20]. In the study of MLLM hallucination, [34] also hypothesizes that certain tokens in MLLM responses aggregate crucial knowledge from contexts, and over-reliance on these tokens can lead to a neglect of the entire image content.

Leading by the common phenomenon of aggregating behaviors, we note that part of the global information in MLLM,

representing visual and textual inputs, is also aggregated into sink tokens. Fig. 5 presents a distribution of cosine similarity between the middle-layer embeddings of multi-modal inputs and the generated tokens. It’s notable that sink tokens, which appear at the turning points of CLIPScore, exhibit a significantly higher resemblance to global input information compared to other tokens. We relate this observation to the hallucinated generation, and make the following analysis.

(1) Attention sinks aggregate information as global context. The aggregating behavior of Transformer-based models is formed naturally during training, with sink tokens receiving high attention scores to aid in subsequent prediction or generation. In Fig. 5, the higher similarity to input embeddings indicates that global multi-modal information is partly integrated into the sink tokens. In the generation process of MLLMs, multi-modal input tokens are positioned before the entire response, serving as a global context. We hypothesize that, inheriting the two-segment response pattern (Section 3.1), attention sinks are chosen to distinguish between segments with different focus, content, and style in MLLM generation. This mechanism provides a more relevant global context for the latter part of the model’s responses, minimizing the need for long-distance attention and aligning with the observed

(a) InstructBLIP

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/eb5de8797af184ae28ff3697e6ae7d34fa5e68db8c984337ca90d8566d6733e7.jpg)

(b) LLaVA-1.5

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/8c9d898ec6e04276b06c7d580d5eaed577c8a98caa267fa707bdaf78ce21b223.jpg)

(c） MiniGPT-4

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-18/bf4fe111-3f0e-41bc-85f9-c3ebe9d5207d/62f96810d09c23313482fafcb84dcad190b6a02ef1122f068962a4972c794307.jpg)

Figure 5: Distribution of cosine similarity between multi-modal input embeddings and generated token embeddings. We compare the similarity of sink tokens (with the postfix -sink) and all other tokens (with the postfix -other).

MLLM generation patterns.

# (2) Misleading aggregation triggers hallucinated response.

While the aggregation process aligns with the generation pattern of MLLMs, we note that only part of the global information is fused into sink tokens, which deviates from the original global information. We speculate that it is still due to deficiencies in the instruction tuning phase, where the second part of responses in training data often includes irrelevant descriptions (Section 3.1), and will mislead the aggregating process with partial, trivial, and even wrong global information. Furthermore, the aggregation of global context into a single token inevitably results in a significant loss of information, diminishing the factual accuracy of the image content. Consequently, MLLMs are trained to aggregate misleading information as context for irrelevant generations. The high attention scores assigned to these sink tokens exacerbate the hallucination problem, introducing irrelevant objects, confused attributes, and incorrect relationships.
