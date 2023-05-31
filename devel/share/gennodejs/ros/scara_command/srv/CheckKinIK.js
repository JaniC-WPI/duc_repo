// Auto-generated. Do not edit!

// (in-package scara_command.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class CheckKinIKRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CheckKinIKRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CheckKinIKRequest
    let len;
    let data = new CheckKinIKRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/CheckKinIKRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CheckKinIKRequest(null);
    return resolved;
    }
};

class CheckKinIKResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.state_q1 = null;
      this.state_q2 = null;
      this.state_q3 = null;
      this.ik_q1 = null;
      this.ik_q2 = null;
      this.ik_q3 = null;
      this.correct = null;
    }
    else {
      if (initObj.hasOwnProperty('state_q1')) {
        this.state_q1 = initObj.state_q1
      }
      else {
        this.state_q1 = 0.0;
      }
      if (initObj.hasOwnProperty('state_q2')) {
        this.state_q2 = initObj.state_q2
      }
      else {
        this.state_q2 = 0.0;
      }
      if (initObj.hasOwnProperty('state_q3')) {
        this.state_q3 = initObj.state_q3
      }
      else {
        this.state_q3 = 0.0;
      }
      if (initObj.hasOwnProperty('ik_q1')) {
        this.ik_q1 = initObj.ik_q1
      }
      else {
        this.ik_q1 = 0.0;
      }
      if (initObj.hasOwnProperty('ik_q2')) {
        this.ik_q2 = initObj.ik_q2
      }
      else {
        this.ik_q2 = 0.0;
      }
      if (initObj.hasOwnProperty('ik_q3')) {
        this.ik_q3 = initObj.ik_q3
      }
      else {
        this.ik_q3 = 0.0;
      }
      if (initObj.hasOwnProperty('correct')) {
        this.correct = initObj.correct
      }
      else {
        this.correct = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CheckKinIKResponse
    // Serialize message field [state_q1]
    bufferOffset = _serializer.float32(obj.state_q1, buffer, bufferOffset);
    // Serialize message field [state_q2]
    bufferOffset = _serializer.float32(obj.state_q2, buffer, bufferOffset);
    // Serialize message field [state_q3]
    bufferOffset = _serializer.float32(obj.state_q3, buffer, bufferOffset);
    // Serialize message field [ik_q1]
    bufferOffset = _serializer.float32(obj.ik_q1, buffer, bufferOffset);
    // Serialize message field [ik_q2]
    bufferOffset = _serializer.float32(obj.ik_q2, buffer, bufferOffset);
    // Serialize message field [ik_q3]
    bufferOffset = _serializer.float32(obj.ik_q3, buffer, bufferOffset);
    // Serialize message field [correct]
    bufferOffset = _serializer.bool(obj.correct, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CheckKinIKResponse
    let len;
    let data = new CheckKinIKResponse(null);
    // Deserialize message field [state_q1]
    data.state_q1 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [state_q2]
    data.state_q2 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [state_q3]
    data.state_q3 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [ik_q1]
    data.ik_q1 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [ik_q2]
    data.ik_q2 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [ik_q3]
    data.ik_q3 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [correct]
    data.correct = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 25;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/CheckKinIKResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '36714075758334348b25a38f8ec94251';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 state_q1
    float32 state_q2
    float32 state_q3
    float32 ik_q1
    float32 ik_q2
    float32 ik_q3
    bool correct
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CheckKinIKResponse(null);
    if (msg.state_q1 !== undefined) {
      resolved.state_q1 = msg.state_q1;
    }
    else {
      resolved.state_q1 = 0.0
    }

    if (msg.state_q2 !== undefined) {
      resolved.state_q2 = msg.state_q2;
    }
    else {
      resolved.state_q2 = 0.0
    }

    if (msg.state_q3 !== undefined) {
      resolved.state_q3 = msg.state_q3;
    }
    else {
      resolved.state_q3 = 0.0
    }

    if (msg.ik_q1 !== undefined) {
      resolved.ik_q1 = msg.ik_q1;
    }
    else {
      resolved.ik_q1 = 0.0
    }

    if (msg.ik_q2 !== undefined) {
      resolved.ik_q2 = msg.ik_q2;
    }
    else {
      resolved.ik_q2 = 0.0
    }

    if (msg.ik_q3 !== undefined) {
      resolved.ik_q3 = msg.ik_q3;
    }
    else {
      resolved.ik_q3 = 0.0
    }

    if (msg.correct !== undefined) {
      resolved.correct = msg.correct;
    }
    else {
      resolved.correct = false
    }

    return resolved;
    }
};

module.exports = {
  Request: CheckKinIKRequest,
  Response: CheckKinIKResponse,
  md5sum() { return '36714075758334348b25a38f8ec94251'; },
  datatype() { return 'scara_command/CheckKinIK'; }
};
