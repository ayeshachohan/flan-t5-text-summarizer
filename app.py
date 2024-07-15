import streamlit as st
from transformers import pipeline, set_seed
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
set_seed(42)

# Load the models with GPU acceleration if available
original_model = pipeline("text2text-generation", model="google/flan-t5-small", device=0)
finetuned_model = pipeline("text2text-generation", model="AyeshaFayyaz/flan-T5_summarizer", device=0)

# Function to generate summaries and calculate ROUGE scores
def generate_summary_and_rouge(prompt, model):
    # Generate summary
    summary = model(prompt, max_length=150, num_beams=4, early_stopping=True)

    # Extract the generated text
    generated_text = summary[0]['generated_text']

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated_text, prompt)

    return generated_text, scores

# Function to plot ROUGE scores
def plot_rouge_scores(original_scores, finetuned_scores):
    labels = ['ROUGE-1', 'ROUGE-L']
    original = [original_scores['rouge1'].fmeasure, original_scores['rougeL'].fmeasure]
    finetuned = [finetuned_scores['rouge1'].fmeasure, finetuned_scores['rougeL'].fmeasure]

    x = range(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, original, width=0.4, label='Original Model', align='center')
    ax.bar(x, finetuned, width=0.4, label='Fine-Tuned Model', align='edge')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylabel('ROUGE F1 Score')
    ax.set_title('ROUGE Score Comparison')

    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Text Summarization Comparison")
    st.markdown("### Original Model (google/flan-t5-small)")
    st.markdown("### Fine-Tuned Model (AyeshaFayyaz/flan-T5_summarizer)")

    # Input prompt from the user
    prompt = st.text_area("Enter your text to summarize:")

    if st.button("Generate Summaries"):
        # Generate summaries and calculate ROUGE scores
        original_summary, original_scores = generate_summary_and_rouge(prompt, original_model)
        finetuned_summary, finetuned_scores = generate_summary_and_rouge(prompt, finetuned_model)

        # Display summaries
        st.subheader("Original Model Summary:")
        st.write(original_summary)
        st.subheader("Fine-Tuned Model Summary:")
        st.write(finetuned_summary)

        # Display ROUGE scores
        st.subheader("ROUGE Scores:")
        st.markdown("#### Original Model:")
        st.write(original_scores)
        st.markdown("#### Fine-Tuned Model:")
        st.write(finetuned_scores)

        # Plot ROUGE scores
        plot_rouge_scores(original_scores, finetuned_scores)

# Run the Streamlit app
if __name__ == "__main__":
    main()
