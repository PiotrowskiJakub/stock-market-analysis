require_relative 'application'
require 'bundler'
Bundler.require(:background)

Sidekiq.configure_client do |config|
  config.redis = {
    # TODO: Tweak this
    size: (ENV['SIDEKIQ_CLIENT_POOL'] || 10).to_i,
    network_timeout: 5
  }
end

Sidekiq.configure_server do |config|
  config.redis = {
    # TODO: Tweak this
    size: (ENV['SIDEKIQ_SERVER_POOL'] || 27).to_i,
    network_timeout: 5
  }
end

Dir.glob('./background/**/*.rb').each { |file| require file }
