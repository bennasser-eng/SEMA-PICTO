import datasets
from datasets import load_dataset , Dataset
from transformers import DataCollatorForTokenClassification

class DataSet():
    def __init__(self):
        super().__init__()

    def prepare_data():
        #Load data
        ds = load_dataset("mnaguib/WikiNER", "fr")

        # to pandas
        df_train = ds["train"].to_pandas()
        df_test = ds["test"].to_pandas()

        # We limit the number of sentences , by a filter
        df_train['count'] = df_train['words'].apply(len)
        df_test['count'] = df_test['words'].apply(len)
        
        df_train = df_train[df_train['count'] < 50]
        df_test = df_test[df_test['count'] <50]

        # We shuffle the data and select a subset of it to speed up the training process
        train_ds = Dataset.from_pandas(df_train)   # len(df_train) = 111803
        train_ds_sampled = train_ds.shuffle(seed=42).select(range(111000))

        train_val = train_ds_sampled.train_test_split(test_size=0.1 ,seed=0)
        df_train = train_val["train"].to_pandas()
        df_val = train_val["test"].to_pandas()

        test_hf_ds = Dataset.from_pandas(df_test)  #len(df_test) = 12589
        test_ds_sampled = test_hf_ds.shuffle(seed=42).select(range(12000))
        df_test = test_ds_sampled.to_pandas()

        # We join the words in a sentence to create a single string for each sentence, which will be used as input for the model
        df_train['sentence'] = df_train['words'].str.join(" ")
        df_test['sentence'] = df_test['words'].str.join(" ")
        df_val['sentence'] = df_val['words'].str.join(" ")  
        return df_train , df_val , df_test
    


    def prepare_datasets(df_train , df_val,df_test, tokenizer): 
        # Conversion en Dataset HF d'abord
        df_train = Dataset.from_pandas(df_train)
        df_val = Dataset.from_pandas(df_val)
        df_test = Dataset.from_pandas(df_test)

        # on passe le tokenizer via fn_kwargs
        train_dataset = df_train.map(
            tokenize_and_align_labels, 
            batched=True,
            batch_size=1000,
            load_from_cache_file=False,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=df_train.column_names,
            num_proc=1
        )


        val_dataset = df_val.map(
            tokenize_and_align_labels, 
            batched=True,
            batch_size=1000,
            load_from_cache_file=False,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=df_val.column_names,
            num_proc=1
            )

        test_dataset = df_test.map(
            tokenize_and_align_labels, 
            batched=True,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=df_test.column_names,
            num_proc=1
        )
        return train_dataset, val_dataset , test_dataset
    

    
    def collator(tokenizer):
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        return data_collator


    @staticmethod
    def tokenize_and_align_labels(examples, tokenizer):
        tokenized_inputs = tokenizer(
            examples["words"].tolist(),
            is_split_into_words=True,
            truncation=True
        )

        all_labels = []
        for i, ner_tags_original in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(ner_tags_original[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

