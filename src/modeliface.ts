import { LlamaContext } from "./llamaEvaluator/LlamaContext.js";
import { LlamaModel, LlamaModelOptions } from "./llamaEvaluator/LlamaModel.js";
import { LLAMAContext, LLAMAModel } from "./utils/getBin.js";

export interface TokenizerInfo {
    nVocab: number;
    eosToken: number;
    bosToken?: number;
    isSpecialToken: boolean[];
    tokenBytes: Uint8Array[];
}

export class ModelIface {
    model: LlamaModel;
    context: LlamaContext;
    _ctx: LLAMAContext;
    _model: LLAMAModel;

    constructor(options: LlamaModelOptions) {
        this.model = new LlamaModel(options);
        this.context = new LlamaContext({ model: this.model });
        this._ctx = (this.context as any)._ctx;
        this._model = this.model._model;
    }

    nextToken(
        tokens: Uint32Array,
        options: {
            temperature?: number;
            tokenMask?: Uint32Array;
            // topK?: number,
            // topP?: number,
            // repeatPenalty?: number,
            // repeatPenaltyTokens?: Uint32Array,
            // repeatPenaltyPresencePenalty?: number,
            // repeatPenaltyFrequencyPenalty?: number,
        }
    ): Promise<number> {
        return this._ctx.eval(tokens, options);
    }

    computeTokenizerInfo(): TokenizerInfo {
        let bosToken: number | undefined = this._ctx.tokenBos();
        if (bosToken < 0) bosToken = undefined;
        const t: TokenizerInfo = {
            nVocab: this._ctx.getVocabSize(),
            eosToken: this._ctx.tokenEos(),
            bosToken,
            tokenBytes: [],
            isSpecialToken: [],
        };
        for (let i = 0; i < t.nVocab; i++) {
            t.isSpecialToken.push(false);
            let bytes = this._ctx.getTokenBytes(i, false);
            if (bytes.length === 0) {
                t.isSpecialToken[i] = true;
                bytes = this._ctx.getTokenBytes(i, true);
            }
            t.tokenBytes.push(bytes);
        }
        return t;
    }

    tokenize(text: string): Uint32Array {
        return this.context.encode(text);
    }
}
