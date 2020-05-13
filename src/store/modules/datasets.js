const state = {
    datasets: [
        {
          "name": "MNIST",
          "path": "data/mnist.csv",
          "description": "A Sample of 400 images of handwritten digits.",
          "appearance": {
            "pointtype": "image",
            "scale": .8,
            "width": 28,
            "height": 28,
            "colorscale": "categorical",
          },
          "sparse": false,
        },
        {
          "name": "VISPUBDATA",
          "path": "data/vispubdata_authors_keywords_refs.csv",
          "description": "Extracted from vispubdata.org. 578 authors, three subspaces. Dimensions 0-611: coauthor-network, 611-1100: keywords, 1100-1480: citation-network.",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "categorical",
          },
          "sparse": true,
        },
        {
          "name": "FMNIST",
          "path": "data/fmnist.csv",
          "description": "A Sample of 768 images of fashion articles.",
          "appearance": {
            "pointtype": "image",
            "scale": .8,
            "width": 28,
            "height": 28,
            "colorscale": "categorical",
          },
          "sparse": false,
        },
        {
          "name": "IRIS",
          "path": "data/iris.csv",
          "description": "Consists of 150 exemplars of the Iris flower.",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "categorical",
          },
          "sparse": false,
        },
        {
          "name": "SPOTIFY",
          "path": "data/spotify.csv",
          "description": "Toy dataset, consists of 400 songs, with features extracted from spotify.",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "categorical",
          },
          "sparse": false,
        },
        {
          "name": "SWISSROLL",
          "path": "data/swissroll.csv",
          "description": "Toy dataset, consists of a swissroll-shaped manifold with 400 points.",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "rainbow",
          },
          "sparse": false,
        },
        {
          "name": "WAVES",
          "description": "Toy dataset, consists of 400 points layed out on a grid, with two mountains and one valley.",
          "path": "data/waves.csv",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "rainbow",
          },
          "sparse": false,
        },
        {
          "name": "S-SHAPE",
          "description": "Toy dataset, consists of a S shaped manifold with 400 points.",
          "path": "data/sshape.csv",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "rainbow",
          },
          "sparse": false,
        },
        {
          "name": "OECD",
          "path": "data/oecd.csv",
          "description": "",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "categorical",
          },
          "sparse": false,
        },
        {
          "name": "IMPOSSIBLE",
          "path": "data/impossible.csv",
          "description": "30 equidistant points.",
          "appearance": {
            "scale": 1,
            "pointtype": "point",
            "colorscale": "single hue",
          },
          "sparse": true,
        },
      ],
}

export default {
    namespaced: true,
    state,
}