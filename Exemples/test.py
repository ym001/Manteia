from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
# see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
