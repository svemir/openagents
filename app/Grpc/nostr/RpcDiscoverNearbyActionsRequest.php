<?php

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: rpc.proto

namespace App\Grpc\nostr;

use Google\Protobuf\Internal\GPBUtil;

/**
 * Generated from protobuf message <code>RpcDiscoverNearbyActionsRequest</code>
 */
class RpcDiscoverNearbyActionsRequest extends \Google\Protobuf\Internal\Message
{
    /**
     * Generated from protobuf field <code>repeated int32 filterByKinds = 1;</code>
     */
    private $filterByKinds;

    /**
     * Generated from protobuf field <code>repeated string filterByNodes = 3;</code>
     */
    private $filterByNodes;

    /**
     * Generated from protobuf field <code>repeated string filterByTags = 4;</code>
     */
    private $filterByTags;

    /**
     * Generated from protobuf field <code>repeated string filterByKindRanges = 5;</code>
     */
    private $filterByKindRanges;

    /**
     * Constructor.
     *
     * @param  array  $data  {
     *                       Optional. Data for populating the Message object.
     *
     * @type array<int>|\Google\Protobuf\Internal\RepeatedField $filterByKinds
     * @type array<string>|\Google\Protobuf\Internal\RepeatedField $filterByNodes
     * @type array<string>|\Google\Protobuf\Internal\RepeatedField $filterByTags
     * @type array<string>|\Google\Protobuf\Internal\RepeatedField $filterByKindRanges
     *                                                             }
     */
    public function __construct($data = null)
    {
        \App\Grpc\nostr\GPBMetadata\Rpc::initOnce();
        parent::__construct($data);
    }

    /**
     * Generated from protobuf field <code>repeated int32 filterByKinds = 1;</code>
     *
     * @return \Google\Protobuf\Internal\RepeatedField
     */
    public function getFilterByKinds()
    {
        return $this->filterByKinds;
    }

    /**
     * Generated from protobuf field <code>repeated int32 filterByKinds = 1;</code>
     *
     * @param  array<int>|\Google\Protobuf\Internal\RepeatedField  $var
     * @return $this
     */
    public function setFilterByKinds($var)
    {
        $arr = GPBUtil::checkRepeatedField($var, \Google\Protobuf\Internal\GPBType::INT32);
        $this->filterByKinds = $arr;

        return $this;
    }

    /**
     * Generated from protobuf field <code>repeated string filterByNodes = 3;</code>
     *
     * @return \Google\Protobuf\Internal\RepeatedField
     */
    public function getFilterByNodes()
    {
        return $this->filterByNodes;
    }

    /**
     * Generated from protobuf field <code>repeated string filterByNodes = 3;</code>
     *
     * @param  array<string>|\Google\Protobuf\Internal\RepeatedField  $var
     * @return $this
     */
    public function setFilterByNodes($var)
    {
        $arr = GPBUtil::checkRepeatedField($var, \Google\Protobuf\Internal\GPBType::STRING);
        $this->filterByNodes = $arr;

        return $this;
    }

    /**
     * Generated from protobuf field <code>repeated string filterByTags = 4;</code>
     *
     * @return \Google\Protobuf\Internal\RepeatedField
     */
    public function getFilterByTags()
    {
        return $this->filterByTags;
    }

    /**
     * Generated from protobuf field <code>repeated string filterByTags = 4;</code>
     *
     * @param  array<string>|\Google\Protobuf\Internal\RepeatedField  $var
     * @return $this
     */
    public function setFilterByTags($var)
    {
        $arr = GPBUtil::checkRepeatedField($var, \Google\Protobuf\Internal\GPBType::STRING);
        $this->filterByTags = $arr;

        return $this;
    }

    /**
     * Generated from protobuf field <code>repeated string filterByKindRanges = 5;</code>
     *
     * @return \Google\Protobuf\Internal\RepeatedField
     */
    public function getFilterByKindRanges()
    {
        return $this->filterByKindRanges;
    }

    /**
     * Generated from protobuf field <code>repeated string filterByKindRanges = 5;</code>
     *
     * @param  array<string>|\Google\Protobuf\Internal\RepeatedField  $var
     * @return $this
     */
    public function setFilterByKindRanges($var)
    {
        $arr = GPBUtil::checkRepeatedField($var, \Google\Protobuf\Internal\GPBType::STRING);
        $this->filterByKindRanges = $arr;

        return $this;
    }
}
