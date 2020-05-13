import Vue from 'vue';
import Vuetify from 'vuetify/lib';

Vue.use(Vuetify);

export default new Vuetify({
  icons: {
    iconfont: 'mdi',
  },
  theme: {
    themes: {
      dark: {
        primary: "#daa520", //"#ff9999", //"#ff6347",
        secondary: "#e8c161",
      },
      light: {
        primary: "#daa520", //"#ff6347",
      }
    }
  }
});
