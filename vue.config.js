const ThreadsPlugin = require('threads-plugin');

module.exports = {
  "transpileDependencies": [
    "vuetify",
  ],
  "configureWebpack": {
      "plugins": [
          new ThreadsPlugin(),
      ],
  },
  "publicPath": process.env.NODE_ENV === "production" ? "/compadre/" : "/",
}