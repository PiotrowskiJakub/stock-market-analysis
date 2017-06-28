Encoding.default_external = Encoding::UTF_8
Encoding.default_internal = Encoding::UTF_8

require 'bundler'
Bundler.setup
Bundler.require(:core, ENV['RACK_ENV'])

require_relative 'env'
require_relative 'database'
Dir.glob('./app/**/*.rb').each { |file| require file }
