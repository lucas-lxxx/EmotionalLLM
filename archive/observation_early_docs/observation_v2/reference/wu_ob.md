# 3 Motivation

The motivation for the untranslation attack is twofold. First, our investigation reveals that current state-of-the-art (SOTA) models benefit from extensive datasets, advanced architectural frameworks, and transfer learning from models pre-trained on large-scale corpora. These factors collectively enhance the models’ robustness in understanding linguistic semantics, thereby complicating the application of traditional semantic attacks within a reasonable perturbation budget. Second, we

observe that contemporary SOTA multilingual speech translation models utilize language-specific tokens as prompts to guide content generation. However, this approach does not guarantee that the output will be in the target language. Despite being directed to produce content in a specified target language, these models inherently exhibit a tendency to generate content in the original source language. Therefore, in this paper, we exploit this property and explore a novel attack approach that misleads the model into outputting content in the source language rather than providing a translation.

Semantic robustness of SOTA models. In this section, we highlight the challenges associated with performing traditional adversarial attacks on SOTA speech translation models through preliminary experiments using the Seamless M4T v2 Large model [10], which contains 2.3 billion parameters and was trained on a large-scale multilingual dataset. For the attack method, we employ the Carlini attack, one of the most widely cited techniques in the automatic speech recognition (ASR) domain. The Carlini attack, which is fundamentally similar to the C&W attack [4], serves as a seminal approach in the realm of ASR adversarial attacks. It lacks additional design elements that enhance imperceptibility and robustness against real-world perturbations, thereby facilitating the optimization of successful adversarial samples.

It is important to note that while some ASR adversarial attacks [17] leverage connectionist temporal classification (CTC) loss to optimize adversarial examples, current SOTA speech translation models do not utilize CTC decoding and instead rely on different loss functions during training. Given that Seamless decodes outputs in an autoregressive manner and is trained using cross-entropy loss [10], we adapt the original Carlini attack by replacing the CTC loss with crossentropy loss.

In this preliminary experiment, our objective is to generate adversarial examples that cause the model to produce a target translation different from the original input. The loss function for this attack is defined as

$$
\mathcal {L} = - \log (p) + \lambda \cdot \| \delta \| _ {2}, \tag {1}
$$

where $p$ is the model output probability for the true token when decoding the first token.

During the attack process, the attacker computes the loss function and employs gradient descent to optimize the input audio waveform. The attack is considered successful when the first token output is altered to a different token. To maintain the original listening experience, the maximum perturbation amplitude is constrained to 0.01. Readers are encouraged to visit https://untranslationattack.github.io/ for an interactive demonstration of the attack. For the optimization, a $\lambda$ of 0.1 is used, and we use the Adam optimizer with a learning rate of 0.001.

We utilize the validated French and German dataset from Common Voice Delta Segment 17.0 [2] for French-to-English

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-15/4f95c18d-a000-4dae-954d-370c89839609/e7a81448526db791020651af4d76fdca7d4ac54bf7954b1814dc14dedfd30041.jpg)

(a) French to English Translation

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-15/4f95c18d-a000-4dae-954d-370c89839609/ac90bf3f503b88566e4d7c8969eb0a4efcef5ea8653c99546992cfb89e7f49e4.jpg)

(b) German to English Translation

Figure 4: Distribution of semantic similarity of output text examples before and after the attack.

and German-to-English speech translation task, randomly selecting 500 samples from each dataset to evaluate the attack. The results of traditional semantic-based attacks are presented in Table 1. It is evident that the attacked translation outputs differ from the original translations, indicating the success of the attack. However, the semantic similarity between the two outputs remains high, and the translation is still comprehensible to users in most cases.

To quantify the impact of these traditional semantic-based attack methods, we used the widely adopted sentence embedding model MiniLM [32]. We converted the translation model’s output text into embeddings with MiniLM and calculated the cosine similarity between the text embeddings before and after the attack as a measure of semantic change. Table 1 also presents the semantic similarity of output text examples before and after the attack, providing a numerical perspective on semantic similarity.

In our experiments, we evaluated the semantic consistency of all samples before and after the attack, and the distribution of semantic similarity is shown in Figure 4. As depicted in the figure, most attacked speech still results in semantically similar translations, with average semantic similarities of 0.7024 and 0.7097 for the two datasets, respectively. This indicates that under reasonable perturbation size constraints, traditional semantic-based attacks can lead to inconsistent outputs. However, the model remains robust in its overall semantic understanding, likely producing different yet semantically similar sentences, without significantly affecting the model’s utility. Further evaluation of traditional attacks is available in Appendix A.

Vulnerability of Untranslation. We further highlight the vulnerability of state-of-the-art (SOTA) speech translation models to untranslation attacks, which is the key motivation of our work. As introduced in Section 2.1, SOTA models employ transformer decoders to generate tokens and control the output language by including a special language token as part of the prompt.

For instance, in the Seamless M4T v2 Large model, the target language token is set as the second token in the output sequence (with the first token being the Begin Of Sequence (BOS) token) before the model decodes the translated text. Despite the strong attention mechanism employed by trans-

Table 1: Demonstrative Results of Traditional Semantic-Based Attacks

<table><tr><td>Original Translation Output</td><td>Attacked Translation Output</td><td>Semantic Similarity</td></tr><tr><td>But the revolution is holding back this development.</td><td>Even the revolution frames the development.</td><td>0.711</td></tr><tr><td>The agency is responsible, throughout the territory, for the public service of welcoming foreigners.</td><td>In all these territories, the agency is responsible for the reception of stranded persons by the public service.</td><td>0.679</td></tr><tr><td>This is in everyone's interest, not mine.</td><td>It's in everyone's best interest, not mine.</td><td>0.826</td></tr><tr><td>Chasing him away from a fugitive who jumps into the ditch.</td><td>He chases him away from a fugitive who jumps into the ditch.</td><td>0.895</td></tr><tr><td>Bonne is located in Danmas.</td><td>Bonn is located in Damascus.</td><td>0.676</td></tr><tr><td>On this occasion, he was made a knight.</td><td>on this occasion, and then the cavalry.</td><td>0.415</td></tr><tr><td>Does your arm hurt you?</td><td>Is your arm hurting?</td><td>0.923</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-15/4f95c18d-a000-4dae-954d-370c89839609/86ff839a4cae114f03721aa38b0343acb482f371559ac3659560ead42c79b581.jpg)

Figure 5: Illustrative token probabilities output by the model when performing ASR and translation tasks. Even when tasked with translation, the model assigns a relatively high probability to the source language token.

former models, which should theoretically enable the model to focus on the target language token and produce a translation in the desired language, our preliminary study found that this token does not consistently guarantee output in the target language.

We use an English-to-French speech translation example, as illustrated in Figure 5. Note that the Transformer Decoder in each row is actually the same model, but with different target language tokens. When the target language token is set to match the source language of the speech, the model operates as an ASR system. The model outputs probabilities for each token in the vocabulary, correctly assigning a high probability to the ground truth token, "Hello", in this case.

When the target language token is set to the intended target language, the model continues to perform accurately, assigning a high probability to the ground truth translation token, "Bonjour". However, it also assigns a relatively high probability to the source language token, "Hello". This suggests a tendency for the model to generate content in the source language even when instructed to produce output in another language. This behavior likely stems from the multi-task learning approach used during training, the inclusion of some corpora that retain the source language for better comprehension, and the paradigm of using language tokens as prompts to guide model output.

To verify the universality of this phenomenon, we con-

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-15/4f95c18d-a000-4dae-954d-370c89839609/3f9230a8c8958fca8c907438ec60ebf47d0cfbd0a56071c79f39a7ff2e4ef553.jpg)

Figure 6: Logit value distribution of specific token during translation. $t o k e n _ { s r c }$ refers to the token output by the model when the language token is set to the source language, while $t o k e n _ { t g t }$ refers to the token output by the model when the language token is set to the target language.

ducted a preliminary experiment to investigate the model’s tendency to output tokens matching the input speech content. Using an English-to-French translation task as an example, we first set the language token to English, the source language, thereby making the model function like an ASR system. We then obtained the token with the highest probability, denoted as $t o k e n _ { \mathrm { s r c } }$ . Next, we set the language token to French, the target language, and obtained the token with the highest probability, denoted as $t o k e n _ { \mathrm { t g t } }$ . For instance, in the illustration in Figure 5 , $t o k e n _ { \mathrm { s r c } }$ is "Hello", and $t o k e n _ { \mathrm { t g t } }$ is "Bonjour". We recorded the logits output by the model during translation for both $t o k e n _ { \mathrm { s r c } }$ and $t o k e n _ { \mathrm { t g t } }$ , as well as the average logits value across all tokens in the vocabulary. The statistical results are presented in Figure 6. A more comprehensive evaluation can be found in Appendix C.

The figure reveals that the model assigns significantly higher logits to $t o k e n _ { \mathrm { t g t } }$ compared to other tokens in most cases, with an average logits value of 15.051 versus 1.092. This suggests that, in practical use, we will not notice any abnormal behavior in model’s operation. However, the model still assigns considerable logits to the tokensrc corresponding to the original phonetic content, though the target language token has been provided as a prompt in translation tasks (7.351 vs. 1.092). This indicates that the model regards outputting the source language tokens as a relatively probable and reasonable option. Furthermore, if an attacker were to target these tokens, the perturbation cost required would be significantly lower than for most other tokens in the vocabulary.
