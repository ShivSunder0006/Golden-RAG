"""
qa_generator.py — Multi-Hop QA Evaluation Set Generator

Creates 5 carefully crafted Question-Answer pairs for RAG evaluation.
At least 3 are multi-hop reasoning questions, at least 1 is cross-video.
All answers are grounded in actual transcript content.
"""


def generate_qa_pairs() -> list[dict]:
    """
    Generate 5 evaluation QA pairs for the RAG pipeline.
    
    Question Types:
        - multi-hop: Requires reasoning across multiple transcript segments
        - cross-video: Combines insights from multiple videos
        - factual: Direct factual recall from a single segment
    
    Returns:
        list[dict]: Each with keys: question, answer, source, question_type
    """
    qa_pairs = [
        # ─── Q1: Multi-hop (Chapter 1 + Chapter 2) — Cross-video ──────────
        {
            "question": (
                "How does the structure of neurons and layers described in Chapter 1 "
                "relate to the cost function and gradient descent process explained in Chapter 2? "
                "Why is the layered architecture essential for gradient descent to work?"
            ),
            "answer": (
                "In Chapter 1, a neural network is structured as layers of neurons where each neuron "
                "holds an activation value between 0 and 1, computed as a weighted sum of previous layer "
                "activations plus a bias, passed through a sigmoid (or ReLU). Chapter 2 explains that "
                "gradient descent minimizes a cost function (average squared error across training examples) "
                "by computing the gradient — partial derivatives of the cost with respect to every weight and bias. "
                "The layered architecture is essential because it creates a differentiable chain from inputs to outputs, "
                "allowing the gradient to be computed layer-by-layer via the chain rule. Without layers, there would be "
                "no compositional structure for backpropagation to decompose the error signal through."
            ),
            "source": "Video 1 (aircAruvnKk) 00:00–06:00 + Video 2 (wjZofJX0v4M) 00:00–08:00",
            "question_type": "multi-hop, cross-video",
        },
        # ─── Q2: Multi-hop (Chapter 3) ────────────────────────────────────
        {
            "question": (
                "In Chapter 3 on backpropagation, how does adjusting a single weight in the last layer "
                "propagate its effect through the cost function? Trace the chain of dependencies from "
                "weight → weighted sum → activation → cost."
            ),
            "answer": (
                "Backpropagation traces the sensitivity of the cost to a weight through a chain: "
                "changing weight w(L) affects the weighted sum z(L), which affects the activation a(L) "
                "through the sigmoid/activation function, which affects the cost C. By the chain rule, "
                "∂C/∂w(L) = ∂z(L)/∂w(L) · ∂a(L)/∂z(L) · ∂C/∂a(L). The first term is the previous layer's "
                "activation a(L-1), the second is the derivative of the activation function σ'(z(L)), "
                "and the third is 2(a(L) - y) for squared error cost. This shows that a weight has more "
                "influence when the corresponding previous-layer neuron is highly active and when the "
                "activation function is sensitive (not saturated)."
            ),
            "source": "Video 3 (fHF22Wxuyw4) 03:00–09:00",
            "question_type": "multi-hop",
        },
        # ─── Q3: Multi-hop (Chapter 1 + Chapter 3) — Cross-video ─────────
        {
            "question": (
                "Chapter 1 describes how neurons detect patterns like edges and loops for digit recognition. "
                "Chapter 3 explains backpropagation as decomposing desired changes layer by layer. "
                "How does backpropagation enable neurons to learn these pattern detectors automatically, "
                "rather than being hand-designed?"
            ),
            "answer": (
                "Chapter 1 presents the intuition that second-layer neurons should detect edges, and third-layer "
                "neurons should detect subcomponents like loops and lines. However, these aren't hand-programmed. "
                "Chapter 3 shows that backpropagation computes how much each weight and bias needs to change to reduce "
                "the cost. When the network sees many examples of, say, the digit 2, the gradient updates naturally "
                "push certain neurons to activate for edges and curves that are consistently present in 2s. "
                "Over thousands of iterations, the weights self-organize so that each neuron becomes a pattern "
                "detector, not because it was designed that way, but because gradient descent finds that configuration "
                "minimizes the cost function. The learned patterns may not always match human intuition — they could "
                "be abstract features that don't correspond to recognizable shapes."
            ),
            "source": "Video 1 (aircAruvnKk) 06:00–12:00 + Video 3 (fHF22Wxuyw4) 00:00–05:00",
            "question_type": "multi-hop, cross-video",
        },
        # ─── Q4: Factual / Conceptual (Chapter 1) ────────────────────────
        {
            "question": (
                "What is the role of the sigmoid function in a neural network as described in Chapter 1, "
                "and why was ReLU mentioned as a modern alternative?"
            ),
            "answer": (
                "The sigmoid function σ(x) = 1/(1+e^(-x)) squashes any real number into the range (0, 1), "
                "which historically was used to represent the 'activation' of a neuron — how much it fires. "
                "It makes very negative values close to 0 and very positive values close to 1. However, "
                "the video notes that modern networks often use ReLU (Rectified Linear Unit), defined as "
                "max(0, x), because it is simpler and avoids the vanishing gradient problem where sigmoid "
                "saturates and makes gradients extremely small in deep networks. ReLU keeps training faster "
                "and more effective in networks with many layers."
            ),
            "source": "Video 1 (aircAruvnKk) 09:00–12:00",
            "question_type": "factual",
        },
        # ─── Q5: Multi-hop (Chapter 2) ────────────────────────────────────
        {
            "question": (
                "In Chapter 2, the concept of gradient descent is visualized as a ball rolling down a hill "
                "in a high-dimensional space. Why does the video emphasize that thinking of the cost function "
                "as having a smooth landscape is misleading, and what does the 'negative gradient direction' "
                "really represent in a space with thousands of weights?"
            ),
            "answer": (
                "The video explains that while the 2D ball-on-a-hill analogy is intuitive, the actual cost "
                "function lives in a space with 13,000+ dimensions (for even a simple network). The gradient "
                "is a vector in this high-dimensional space whose components tell you the relative sensitivity "
                "of the cost to each weight. The negative gradient direction is the direction of steepest descent "
                "in this space, but the landscape is not a smooth bowl — it has many local minima. The video "
                "emphasizes that each component of the gradient encodes not just the direction but also the "
                "magnitude of sensitivity: weights connected to brighter neurons or with larger activations "
                "have larger gradient components, meaning they get adjusted more. The learning rate then scales "
                "how big a step you take along this direction."
            ),
            "source": "Video 2 (wjZofJX0v4M) 05:00–14:00",
            "question_type": "multi-hop",
        },
    ]

    return qa_pairs


def get_qa_table_data() -> list[list[str]]:
    """Return QA pairs formatted as rows for a table display."""
    qa_pairs = generate_qa_pairs()
    return [
        [qa["question"], qa["answer"], qa["source"], qa["question_type"]]
        for qa in qa_pairs
    ]


def get_methodology() -> str:
    """
    Return the methodology section describing how QA pairs were selected
    and what they test in a RAG pipeline.
    """
    return """
## Methodology

### Question Selection Criteria

Questions were designed to evaluate three core RAG capabilities:

1. **Retrieval Accuracy** — Can the system find the right transcript chunks?
   - Factual questions (Q4) test basic retrieval: the answer exists in a single contiguous segment.
   - A system that embeds chunks well and performs accurate cosine similarity should retrieve the right passage.

2. **Multi-Hop Reasoning** — Can the system combine information from multiple chunks?
   - Questions Q1, Q2, Q3, Q5 require synthesizing information across non-adjacent transcript segments.
   - Q2 traces a chain of mathematical dependencies (weight → z → activation → cost), requiring the system to link concepts that appear in different parts of the same video.
   - Q5 requires connecting the intuitive analogy (ball rolling down a hill) with the formal mathematical description of gradient components.

3. **Cross-Source Synthesis** — Can the system integrate knowledge across videos?
   - Q1 bridges Chapter 1's network structure with Chapter 2's gradient descent — different videos, complementary concepts.
   - Q3 connects Chapter 1's pattern detection intuition with Chapter 3's backpropagation mechanics to explain emergent learning.

### Transcript Processing

- Transcripts were fetched via `youtube-transcript-api` with fallback across manual English, auto-generated English, and Hindi transcripts.
- Text segments were merged into continuous passages, preserving timestamps for source attribution.
- Punctuation and formatting artifacts from auto-generated captions were cleaned.

### Failure Cases — What Weak RAG Systems Get Wrong

| Failure Mode | Description | Affected Questions |
|---|---|---|
| **Chunk boundary misses** | If chunk splits fall mid-explanation, the system retrieves incomplete context. Q2's chain rule derivation spans ~6 minutes. | Q2, Q5 |
| **Single-hop retrieval** | System retrieves one relevant chunk but misses the second required chunk. Cross-video questions become impossible. | Q1, Q3 |
| **Keyword over-reliance** | Searching for "sigmoid" finds Chapter 1 mentions but misses the Chapter 3 context about gradient saturation. | Q4 (partially), Q2 |
| **Hallucination patterns** | System invents plausible-sounding math (e.g., wrong chain rule terms) or attributes concepts to the wrong video. | Q2, Q5 |
| **Language confusion** | Hindi/English mixed transcripts cause embedding misalignment; bilingual content gets lower similarity scores. | All (if Video 4 is involved) |

### Hallucination Risk Examples

- A weak system might answer Q2 by stating the correct chain rule formula but invent the specific terms (e.g., claiming ∂C/∂w involves the learning rate directly).
- For Q1, a system might correctly describe layers but fabricate the connection to gradient descent rather than retrieving it from Chapter 2.
- For Q5, a system might describe gradient descent generically from training data rather than citing the specific 13,000-dimension framing from the video.
"""


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📋 QA Evaluation Set\n")
    for i, qa in enumerate(generate_qa_pairs(), 1):
        print(f"Q{i} [{qa['question_type']}]: {qa['question'][:80]}...")
        print(f"  Source: {qa['source']}\n")

    print("\n📝 Methodology Preview:")
    print(get_methodology()[:500])
