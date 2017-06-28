require 'bundler'
Bundler.require(:web)
require_relative 'application'
require_relative 'background'
Oj.default_options = { mode: :custom, use_to_hash: true, symbol_keys: true, bigdecimal_load: true }

Dir.glob('./web/**/*.rb').each { |file| require file }
