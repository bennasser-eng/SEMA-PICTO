import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel , AutoConfig
from MLP import MLP
from dataset import DataSet
import torch
import torch.nn as nn




class MLP(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(MLP, self).__init__()
    self.classifier_head = nn.Sequential(
       nn.Linear(input_dim, 1024),
       nn.ReLU(),
       nn.Dropout(0.3),
       nn.Linear(1024, 512),
       nn.ReLU(),
       nn.Dropout(0.3),
       nn.Linear(512, 256),
       nn.ReLU(),
       nn.Linear(256, 128),
       nn.ReLU(),
       nn.Dropout(0.4),
       nn.Linear(128, 64),
       nn.ReLU(),
       nn.Dropout(0.3),
       nn.Linear(64, output_dim)    # Output directly num_labels logits = 5
    )

  def forward(self, x):
    return self.classifier_head(x)



class NER_models(nn.Module):
    def __init__(self, model_name, num_labels, token=None):
        super(NER_models, self).__init__()
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=token)

        if not hasattr(config, "model_type") or config.model_type != "text":
            config.model_type = "text"

        self.encoder = AutoModel.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto",
                                                 token=token,
                                                 trust_remote_code=True,
                                                 )
        
       # self.encoder.gradient_checkpointing_enable()        
        # on gele les couches de encoder par defaut
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Dégèle sélectif
        layers = None
        
        # Cas XLM-RoBERTa
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
        # Cas  CamemBERT / Pantagruel
        elif hasattr(self.encoder, "layer"):
            layers = self.encoder.layer

        if layers:
            num_total_layers = len(layers)
            # On dégel les 2 dernières couches
            for i in range(num_total_layers - 2, num_total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
                print(f"[{model_name}] Layer {i} unfrozen.")

        # MLP 
        self.classifier = MLP(self.encoder.config.hidden_size, num_labels)
        self.classifier = self.classifier.to(torch.float16)
        self.num_labels = num_labels    # We store num_labels for loss calculation
 
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Explicitly pass mode='text' for text inputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]    # We take the last hidden state

        # classifier(MLP)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
   
