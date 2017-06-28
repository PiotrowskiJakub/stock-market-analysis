module RubyApp
  class APIv1 < Grape::API
    version 'v1', using: :path
    content_type :csv, 'text/csv'
    content_type :json, 'application/json'
    default_format :json
    formatter :json, ->(object, env) { Oj.dump(object) }
    prefix :api

  class API < Grape::API
    mount RubyApp::APIv1
  end
end
