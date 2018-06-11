import Vue from 'vue'
import VueRouter from 'vue-router';
import App from '@/App'
import routes from '@/routes'
import 'bulma/bulma.sass'

Vue.use(VueRouter);

new Vue({
  el: '#app',
  render: h => h(App),
  router: new VueRouter({ routes })
})
