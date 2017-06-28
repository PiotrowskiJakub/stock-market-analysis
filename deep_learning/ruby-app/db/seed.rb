require_relative '../config/application'

buckets = [
  Buckets::UGC::Other::NoChannelId,
  Buckets::UGC::Other::ChannelMatch,
  Buckets::UGC::Other::MultipleClaims,
  Buckets::UGC::DescriptiveSearch::NoChannelId,
  Buckets::UGC::DescriptiveSearch::ChannelMatch,
  Buckets::UGC::DescriptiveSearch::MultipleClaims,
  Buckets::UGC::ContentID::NoChannelId,
  Buckets::UGC::ContentID::ChannelMatch,
  Buckets::UGC::ContentID::MultipleClaims,
  Buckets::PartnerProvided::NoChannelId,
  Buckets::PartnerProvided::ChannelMatch,
  Buckets::PartnerProvided::MultipleClaims
]

buckets.each do |bucket_class|
  bucket = bucket_class.new
  bucket.save if bucket.valid?
end
