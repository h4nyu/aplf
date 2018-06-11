import Error404 from "@/pages/Error404";
import Home from "@/pages/Home";
export default [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '*',
    name: 'error404',
    component: Error404
  }
];
