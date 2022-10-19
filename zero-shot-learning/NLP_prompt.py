from transformers import AutoModelForMaskedLM , AutoTokenizer
import torch

class Prompting(object):
    """ doc string 
    This class helps us to implement
    Prompt-based Learning Model
    """
    def __init__(self, **kwargs):
        """ constructor 
        parameter:
        ----------
           model: AutoModelForMaskedLM
                path to a Pre-trained language model form HuggingFace Hub
           tokenizer: AutoTokenizer
                path to tokenizer if different tokenizer is used, 
                otherwise leave it empty
        """
        model_path=kwargs['model']
        
        if "tokenizer" in kwargs.keys():
            tokenizer_path= kwargs['tokenizer']
        else:
            tokenizer_path= kwargs['model']
            
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
 
    def compute_max_seq_len(self) -> int:
        """Compute the maximum sequence length of an input sequence.

        Large input sequences need to be truncated while keeping the
        prompt template untruncated.

        Returns:
            The maximum sequence length in number of tokens.
        """
        
        template_len=len(self.tokenizer(self.prompt,add_special_tokens=False).input_ids)
        
        n_special_chars = self.tokenizer.num_special_tokens_to_add(False)

        max_seq_len = self.tokenizer.model_max_length \
            - (template_len + n_special_chars)

        assert max_seq_len > 0

        return max_seq_len

    def prompt_pred(self,text,device):
        """
          Predict MASK token by listing the probability of candidate tokens 
          where the first token is the most likely
          Parameters:
          ----------
          text: str 
              The text including [MASK] token.
              It only supports single MASK token. If more [MASK]ed tokens 
              are given, it takes the first one.
          Returns:
          --------
          list of (token, prob)
             The return is a list of all token in LM Vocab along with 
             their prob score, sort by score in descending order 
        """
        indexed_tokens=self.tokenizer(text, return_tensors="pt").input_ids
        if indexed_tokens.shape[1]>self.tokenizer.model_max_length:
            x1=indexed_tokens[0,0:3]  ### "[CLS] email :"
            x2=indexed_tokens[0,-7:] ### ". this email has [MASK] sentiment [SEP]"
            x3=indexed_tokens[0,3:self.tokenizer.model_max_length-10]
            indexed_tokens=torch.cat([x1,x3,x2]).unsqueeze(0)
            
        tokenized_text= self.tokenizer.convert_ids_to_tokens(indexed_tokens[0])
        # take the first masked token
        mask_pos=tokenized_text.index(self.tokenizer.mask_token)
        self.model.eval()
        with torch.no_grad():
            self.model=self.model.to(device)
            indexed_tokens=indexed_tokens.to(device)
            outputs = self.model(indexed_tokens)
            predictions = outputs[0]
        values, indices=torch.sort(predictions[0, mask_pos],  descending=True)
        #values=torch.nn.functional.softmax(values, dim=0)
        result=list(zip(self.tokenizer.convert_ids_to_tokens(indices), values))
        self.scores_dict={a:b for a,b in result}
        return result

    def compute_tokens_prob(self, text, token_list1, token_list2, device):
        """
        Compute the activations for given two token list, 
        Parameters:
        ---------
        token_list1: List(str)
         it is a list for positive polarity tokens such as good, great. 
        token_list2: List(str)
         it is a list for negative polarity tokens such as bad, terrible.      
        Returns:
        --------
        Tuple (
           the probability for first token list,
           the probability of the second token list,
           the ratio score1/ (score1+score2)
           The softmax returns
           )
        """
        _=self.prompt_pred(text,device)
        score1=[self.scores_dict[token1] if token1 in self.scores_dict.keys() else 0\
                for token1 in token_list1]
        score1= sum(score1)
        score2=[self.scores_dict[token2] if token2 in self.scores_dict.keys() else 0\
                for token2 in token_list2]
        score2= sum(score2)
        
        softmax_rt=torch.nn.functional.softmax(torch.Tensor([score1,score2]), dim=0)
        return softmax_rt        
    
    def fine_tune(self, sentences, labels, prompt="it was [MASK] sentiment.",goodToken="positive",badToken="negative"):
        """  
          Fine tune the model
        """
        good=tokenizer.convert_tokens_to_ids(goodToken)
        bad=tokenizer.convert_tokens_to_ids(badToken)

        from transformers import AdamW
        optimizer = AdamW(self.model.parameters(),lr=1e-3)

        for sen, label in zip(sentences, labels):
            tokenized_text = self.tokenizer.tokenize(sen+prompt)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            # take the first masked token
            mask_pos=tokenized_text.index(self.tokenizer.mask_token)
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
            pred=predictions[0, mask_pos][[good,bad]]
            prob=torch.nn.functional.softmax(pred, dim=0)
            lossFunc = torch.nn.CrossEntropyLoss()
            loss=lossFunc(prob.unsqueeze(0), torch.tensor([label]))
            loss.backward()
            optimizer.step()
        print("done!")
        
        