const path = require('path');
const setPath = function(folderName) {
  return path.join(__dirname, folderName);
}
const NODE_ENV = process.env.NODE_ENV;
const config = {
  mode: NODE_ENV,
  entry: './src/index.js',
  module: {
    rules: [
      {
        test: /\.js$/,
        use: [
          {
            loader: 'babel-loader',
            options: {
              presets: [
                ['env', {'modules': false}]
              ]
            }
          }
        ]
      }
    ]
  }
};
module.exports = config;
