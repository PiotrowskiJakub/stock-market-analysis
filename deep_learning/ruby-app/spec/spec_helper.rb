ENV['RACK_ENV'] = 'test'

require 'bundler'
require 'webmock/rspec'

require_relative '../config/application'

Dir['./spec/support/**/*.rb'].each { |f| require f }

VCR.configure do |config|
  config.cassette_library_dir = 'spec/fixtures/vcr_cassettes'
  config.hook_into :webmock
end

RSpec.configure do |config|
  config.formatter = 'NyanCatFormatter'
  config.include FactoryGirl::Syntax::Methods
  config.include RspecSequel::Matchers

  config.before(:suite) do
    FactoryGirl.find_definitions
  end

  config.around(:each) do |example|
    DB.transaction(rollback: :always, auto_savepoint: true) { example.run }
  end
end
