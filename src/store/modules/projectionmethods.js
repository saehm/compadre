const state = {
    methods: [
        {
            "name": "UMAP",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [
                {
                    "name": "local_connectivity",
                    "type": "number",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "selection": [1, 1],
                },
                {
                    "name": "min_dist",
                    "type": "number",
                    "min": 0,
                    "max": 2,
                    "step": .1,
                    "selection": [1, 1],
                },
            ],
        },
        {
            "name": "tSNE",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [
                {
                    "name": "perplexity",
                    "type": "number",
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "selection": [50, 50],
                },
                {
                    "name": "epsilon",
                    "type": "number",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "selection": [5, 5],
                },
            ],
        },
        {
            "name": "TriMap",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [
                {
                    "name": "weight_adj",
                    "type": "number",
                    "min": 100,
                    "max": 10000,
                    "step": 100,
                    "selection": [500, 500],
                },
                {
                    "name": "c",
                    "type": "number",
                    "min": 1,
                    "max": 10,
                    "std": 5,
                    "step": 1,
                    "selection": [5, 5],
                },
            ],
        },
        {
            "name": "MDS",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [],
        },
        {
            "name": "PCA",
            "seed": 1212,
            "parameters": [],
        },
        {
            "name": "ISOMAP",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [
                {
                    "name": "neighbors",
                    "type": "number",
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "selection": [15, 17],
                },
            ],
        },
        {
            "name": "LLE",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [
                {
                    "name": "neighbors",
                    "type": "number",
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "selection": [15, 17],
                }
            ],
        },
        {
            "name": "LTSA",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [
                {
                    "name": "neighbors",
                    "type": "number",
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "selection": [15, 17],
                },
            ],
        },
        /* {
            "name": "LDA",
            "seed": 1212,
            "metric": "euclidean",
            "parameters": [
                {
                    "name": "labels",
                    "type": "array",
                },
            ],
        }, */
    ]
}

export default {
    namespaced: true,
    state,
}