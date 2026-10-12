# Sequence Generation
import numpy as np
import math
from collections import defaultdict


task_choices = ["var-copy", "var-copy-rep", "decode-recall", "decode-recall-last", "assoc-recall", "assoc-recall-mk", "needle", "mqar", "mkar", "fuzzy", "flip"]

# Associative-recall variants (the "mqar"/"mkar"/"fuzzy" branch below).
MQAR_VALUE_LEN = 2   # mqar: 1-token key, multi-token value
MKAR_KEY_LEN = 2     # mkar: multi-token key, 1-token value
FUZZY_KEY_LEN = 2    # fuzzy: multi-token key and value
FUZZY_VALUE_LEN = 2

# "flip" counting task (from "Knee Deep in C-RASP"): the label turns 1 after FLIP_K flips.
FLIP_K = 6

def force_args(args):
    if args.train_task in ["var-copy", "var-copy-rep"]:
        pass

    if args.train_task in ["decode-recall", "decode-recall-last"]:
        args.num_numbers = 2
        args.num_vocab = int(2 ** math.floor(math.log(args.num_vocab) / math.log(2)))

    if args.train_task == "assoc-recall":
        args.num_numbers = 0

    if args.train_task == "assoc-recall-mk":
        args.num_numbers = 0
        # args.num_vocab = 1 + int(args.num_vocab ** (1./size_key))

    if args.train_task == "needle":
        args.num_numbers = 2

    if args.train_task in ["mqar", "mkar", "fuzzy"]:
        args.num_numbers = 1  # #0 separates key and value

    if args.train_task == "flip":
        args.num_numbers = 2  # labels #0 / #1; vocab stays 2


def generate_seq(tokenizer, length, task, p=0.2, mixed=False):
    num_vocab = tokenizer.num_vocab
    num_numbers = tokenizer.num_numbers
    
    if task == "var-copy":
        # Start with num_numbers vocab tokens
        if mixed:
            if np.random.rand() < 0.5:
                # Hard for the TF
                input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.03) 
            else:
                # Hard for the SSM
                input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.2) 
                input_seq[-1] = np.random.choice(tokenizer.number_tokens)
        else:
            input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        # input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers) 
        
        nums = [(i, int(c[1:])) for (i, c) in enumerate(input_seq) if c in tokenizer.number_tokens]
        
        # The real task, if not degenerate
        if len(nums) > 0:
            output_seq = ["<null>"] * nums[0][0]

            for i in range(len(nums)-1):
                if nums[i][0]-nums[i][1] < 0:
                    if nums[i+1][0]-nums[i][1] < 0:
                        output_seq += ["<null>"] * (nums[i+1][0]-nums[i][0])
                    else:
                        output_seq += ["<null>"] * (nums[i][1]-nums[i][0])
                        output_seq += input_seq[:nums[i+1][0]-nums[i][1]]
                else:
                    output_seq += input_seq[nums[i][0]-nums[i][1]:nums[i+1][0]-nums[i][1]]

            if nums[-1][0]-nums[-1][1] < 0:
                output_seq += ["<null>"] * (nums[-1][1]-nums[-1][0])
                output_seq += input_seq[:-nums[-1][1]]
            else:
                output_seq += input_seq[nums[-1][0]-nums[-1][1]:-nums[-1][1]]
        else:
            output_seq = ["<null>"] * length

        # output_seq = ["<bos>"] + output_seq[:length] + ["<eos>"]

    elif task == "var-copy-rep":
        # Start with num_numbers vocab tokens
        # input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        input_seq = rand_seq_special(tokenizer, length, num_vocab, num_numbers, p_numbers=p, special_type="repetitive_vocab") 
        
        nums = [(i, int(c[1:])) for (i, c) in enumerate(input_seq) if c in tokenizer.number_tokens]
        
        # The real task, if not degenerate
        if len(nums) > 0:
            output_seq = ["<null>"] * nums[0][0]

            for i in range(len(nums)-1):
                if nums[i][0]-nums[i][1] < 0:
                    if nums[i+1][0]-nums[i][1] < 0:
                        output_seq += ["<null>"] * (nums[i+1][0]-nums[i][0])
                    else:
                        output_seq += ["<null>"] * (nums[i][1]-nums[i][0])
                        output_seq += input_seq[:nums[i+1][0]-nums[i][1]]
                else:
                    output_seq += input_seq[nums[i][0]-nums[i][1]:nums[i+1][0]-nums[i][1]]

            if nums[-1][0]-nums[-1][1] < 0:
                output_seq += ["<null>"] * (nums[-1][1]-nums[-1][0])
                output_seq += input_seq[:-nums[-1][1]]
            else:
                output_seq += input_seq[nums[-1][0]-nums[-1][1]:-nums[-1][1]]
        else:
            output_seq = ["<null>"] * length

        input_seq = ["<bos>"] + input_seq + ["<eos>"]
        output_seq = ["<bos>"] + output_seq + ["<eos>"]
        # output_seq = ["<bos>"] + output_seq[:length] + ["<eos>"]

    elif task == "decode-recall":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        output_seq = [None for _ in range(len(input_seq))]

        assoc = {v: "<null>" for v in tokenizer.vocab}
        s = 0
        for i in range(len(output_seq)):
            if i != 0:
                assoc[input_seq[i-1]] = input_seq[i]
            
            if input_seq[i][0] == '#':
                # s = (2 * s + int(input_seq[i][1:])) % num_numbers
                s = (2 * s + int(input_seq[i][1:])) % num_vocab

            # if i-s < 0:
            #     output_seq[i] = "<null>"
            # else:
            #     output_seq[i] = input_seq[i-s]

            output_seq[i] = assoc["V%d" % s]

    elif task == "decode-recall-last":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        n_bits = int(math.log(num_vocab)/math.log(2))

        target = np.random.randint(0, num_vocab)
        temp = target
        for i in range(length-1, length-1-n_bits, -1):
            input_seq[i] = "#%d" % (temp % 2)
            temp = temp // 2

        try:
            i = length-2-n_bits - input_seq[-2-n_bits::-1].index("V%d" % target)
            output_seq[-1] = input_seq[i+1]
        except ValueError:
            pass

    elif task == "assoc-recall":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.2) 
        output_seq = [None for _ in range(len(input_seq))]

        assoc = {v: "<null>" for v in tokenizer.vocab}

        for i in range(len(output_seq)):
            if i != 0:
                assoc[input_seq[i-1]] = input_seq[i]

            output_seq[i] = assoc[input_seq[i]]

    elif task == "assoc-recall-mk":
        size_key = 2
        
        input_seq = rand_seq(tokenizer, length, num_vocab, 0, p_numbers=0.0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        assoc = defaultdict(lambda: "<null>")

        for i in range(len(output_seq)):
            if i > size_key:
                key = tuple(input_seq[i-size_key:i])
                assoc[key] = input_seq[i]

            if i+1 > size_key:
                key = tuple(input_seq[i-size_key+1:i+1])
                output_seq[i] = assoc[key]

    elif task == "needle":
        needle_length = 1
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0) 
        input_seq[-needle_length:] = ["#1" for _ in range(needle_length)]

        # Needle position
        needle_pos = np.random.randint(0, length // 2)
        input_seq[needle_pos] = "#0"
        needle = input_seq[needle_pos + 1:needle_pos + needle_length + 1]

        # Ask for the needle at the end
        output_seq = ["<null>" for _ in range(len(input_seq))]
        output_seq[needle_pos:] = [needle[-1] for i in range(len(input_seq) - needle_pos)]
        # output_seq[-needle_length:] = needle

    elif task in ("mqar", "mkar", "fuzzy"):
        # Items: <bos> <key...> #0 <value...> <eos>
        #   mqar:  key_len=1, value_len>1
        #   mkar:  key_len>1, value_len=1
        #   fuzzy: key_len>1, value_len>1
        # The first occurrence of a key defines it (value shown, output <null>). A later
        # occurrence is a query: the input value span is <null> and the output holds the
        # stored value. ce_loss is unshifted, so targets sit on the value positions and
        # accuracy counts only query value spans. Keys are never redefined.
        if task == "mqar":
            key_len, val_len = 1, MQAR_VALUE_LEN
        elif task == "mkar":
            key_len, val_len = MKAR_KEY_LEN, 1
        else:  # fuzzy
            key_len, val_len = FUZZY_KEY_LEN, FUZZY_VALUE_LEN
        item_len = 1 + key_len + 1 + val_len + 1  # bos, key, sep, value, eos
        sep = "#0"

        store = {}  # key tuple -> list of value tokens

        def rand_key():
            for _ in range(50):
                k = tuple("V%d" % np.random.randint(num_vocab) for _ in range(key_len))
                if k not in store:
                    return k
            return None

        input_seq, output_seq = [], []
        while len(input_seq) + item_len <= length:
            if store and np.random.rand() < 0.5:                 # QUERY
                key = list(store.keys())[np.random.randint(len(store))]
                in_vals, out_vals = ["<null>"] * val_len, list(store[key])
            else:                                                # DEFINITION
                key = rand_key()
                if key is None:
                    break
                value = ["V%d" % np.random.randint(num_vocab) for _ in range(val_len)]
                store[key] = value
                in_vals, out_vals = list(value), ["<null>"] * val_len
            input_seq  += ["<bos>"] + list(key) + [sep] + in_vals + ["<eos>"]
            output_seq += ["<null>"] * (2 + key_len) + out_vals + ["<null>"]

        pad = length - len(input_seq)
        input_seq  += ["<null>"] * pad
        output_seq += ["<null>"] * pad

    elif task == "flip":
        # Flip counting: vocab V0/V1, a flip is two adjacent tokens that differ.
        # The label is #0 until the FLIP_K-th flip, then #1. Example, k=6:
        #   in : V0 V1 V1 V0 V1 V0 V0 V1 V0 V1
        #   out:  0  0  0  0  0  0  0  0  0  1
        # Every position is scored, so compare accuracy with the majority-class rate.
        input_seq = ["V%d" % np.random.randint(num_vocab) for _ in range(length)]
        output_seq = []
        flips = 0
        for i in range(length):
            if i > 0 and input_seq[i] != input_seq[i - 1]:
                flips += 1
            output_seq.append("#1" if flips >= FLIP_K else "#0")

    else:
        print("Task name:", task)
        assert False # Not implemented

    return input_seq, output_seq

################################################################################################

# SEQUENCE GENERATION HELPERS

def rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=-1):
    if p_numbers == -1:
        p_numbers = num_numbers / (num_vocab + num_numbers)

    if num_numbers != 0:
        props = {"V": (1-p_numbers)/num_vocab, "#": p_numbers/num_numbers, "<": 0}  
    else:
        props = {"V": 1/num_vocab, "#": 0, "<": 0}  
    props = np.array([props[i[0]] for i in tokenizer.vocab])

    return np.random.choice(tokenizer.vocab, size=length, p=props).tolist()


# For other special generations
def rand_seq_special(tokenizer, length, num_vocab, num_numbers, p_numbers=-1, special_type=None):
    if special_type == "repetitive_vocab":
        if p_numbers == -1:
            p_numbers = num_numbers / (num_vocab + num_numbers)
    
        if num_numbers != 0:
            props = {"V": 0, "#": p_numbers/num_numbers, "<": 0} 
        else:
            props = {"V": 0, "#": 0, "<": 0}  
        if num_numbers != 0:
            props_V0 = (1-p_numbers)
        else:
            props_V0 = 1
        props = np.array([props[i[0]] if i != "V0" else props_V0 for i in tokenizer.vocab])
        
        tile_length = 3

        props_tile = {"V": 1./num_vocab, "#": 0, "<": 0} 
        props_tile = np.array([props_tile[i[0]] for i in tokenizer.vocab])

        ret_seq = np.random.choice(tokenizer.vocab, size=length, p=props)
        ret_seq2 = np.tile(np.random.choice(tokenizer.vocab, size=tile_length, p=props_tile), (length // tile_length + 1))[:length]

        return np.where(ret_seq == "V0", ret_seq2, ret_seq).tolist()

    else:
        assert False, "Not implemented"