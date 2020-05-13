import Vue from 'vue'
import App from './App.vue'
import './registerServiceWorker'
import store from './store'
import vuetify from './plugins/vuetify';
import VueObserveVisibility from "vue-observe-visibility";

Vue.config.productionTip = false
Vue.use(VueObserveVisibility);

new Vue({
  store,
  vuetify,
  render: h => h(App)
}).$mount('#app')
