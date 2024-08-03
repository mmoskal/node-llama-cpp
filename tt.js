import { LlamaModel, LlamaContext, LlamaChatSession } from "node-llama-cpp";



async function tt() {

    const model = new LlamaModel({
        modelPath: "test/.models/Phi-3.1-mini-4k-instruct-Q5_K_M.gguf"
    });
    const context = new LlamaContext({ model });

    let toks = context.encode("Hi there, how are you?");
    console.log(toks)
    console.log(context.decode(toks));
    console.log(context.decode([131]));
    console.log(context._ctx.getTokenString(131));
    console.log(context._ctx.getTokenBytes(131));
    console.log(context._ctx.getTokenBytes(131, true));

    console.log(context._ctx.getTokenBytes(32000));
    console.log(context._ctx.getTokenBytes(32000, true));
    console.log(context._ctx.getVocabSize());

    let t0 = Date.now();
    let tokens = []
    let n_vocab = context._ctx.getVocabSize();
    for (let i = 0; i < n_vocab; i++) {
        tokens.push(context._ctx.getTokenBytes(i));
    }
    console.log("Time: ", Date.now() - t0);

    toks = context.encode(Uint8Array.from([128, 129]));
    console.log(toks)

    const q1 = "Q: Hi there, how are you?\nA:";
    toks = context.encode(q1);
    const mask_size = (n_vocab + 31) >> 5;
    for (let i = 0; i < 10; ++i) {
        const tokenMask = new Uint32Array(mask_size);
        tokenMask.fill(0xffffffff);
        const skip = 306;
        // tokenMask[skip >> 5] &= ~(1 << (skip & 31));
        const r = await context._ctx.eval(toks, { tokenMask });
        toks = Uint32Array.from([...toks, r]);
        console.log({ tok: r, str: context._ctx.getTokenString(r) });
    }
}


tt()

// const session = new LlamaChatSession({context});

// const q1 = "Hi there, how are you?";
// console.log("User: " + q1);

// const a1 = await session.prompt(q1);
// console.log("AI: " + a1);


// const q2 = "Summarize what you said";
// console.log("User: " + q2);

// const a2 = await session.prompt(q2);
// console.log("AI: " + a2);