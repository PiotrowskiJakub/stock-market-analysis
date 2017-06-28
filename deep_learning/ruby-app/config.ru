require_relative 'config/web'
require 'sidekiq/web'

app_api = Rack::Builder.new do
  run RubyApp::API
end

sidekiq = Rack::Builder.new do
  use Rack::Auth::Basic, 'Protected Area' do |username, password|
    Rack::Utils.secure_compare(Digest::SHA256.hexdigest(username), Digest::SHA256.hexdigest(ENV['SIDEKIQ_USERNAME'])) &
      Rack::Utils.secure_compare(Digest::SHA256.hexdigest(password), Digest::SHA256.hexdigest(ENV['SIDEKIQ_PASSWORD']))
  end

  run Sidekiq::Web
end

run Rack::URLMap.new('/' => app_api, '/sidekiq' => sidekiq)
