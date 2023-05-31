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

class CheckKinFKRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CheckKinFKRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CheckKinFKRequest
    let len;
    let data = new CheckKinFKRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/CheckKinFKRequest';
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
    const resolved = new CheckKinFKRequest(null);
    return resolved;
    }
};

class CheckKinFKResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.state_x = null;
      this.state_y = null;
      this.state_z = null;
      this.state_phi = null;
      this.state_theta = null;
      this.state_psi = null;
      this.fk_x = null;
      this.fk_y = null;
      this.fk_z = null;
      this.fk_phi = null;
      this.fk_theta = null;
      this.fk_psi = null;
      this.correct = null;
    }
    else {
      if (initObj.hasOwnProperty('state_x')) {
        this.state_x = initObj.state_x
      }
      else {
        this.state_x = 0.0;
      }
      if (initObj.hasOwnProperty('state_y')) {
        this.state_y = initObj.state_y
      }
      else {
        this.state_y = 0.0;
      }
      if (initObj.hasOwnProperty('state_z')) {
        this.state_z = initObj.state_z
      }
      else {
        this.state_z = 0.0;
      }
      if (initObj.hasOwnProperty('state_phi')) {
        this.state_phi = initObj.state_phi
      }
      else {
        this.state_phi = 0.0;
      }
      if (initObj.hasOwnProperty('state_theta')) {
        this.state_theta = initObj.state_theta
      }
      else {
        this.state_theta = 0.0;
      }
      if (initObj.hasOwnProperty('state_psi')) {
        this.state_psi = initObj.state_psi
      }
      else {
        this.state_psi = 0.0;
      }
      if (initObj.hasOwnProperty('fk_x')) {
        this.fk_x = initObj.fk_x
      }
      else {
        this.fk_x = 0.0;
      }
      if (initObj.hasOwnProperty('fk_y')) {
        this.fk_y = initObj.fk_y
      }
      else {
        this.fk_y = 0.0;
      }
      if (initObj.hasOwnProperty('fk_z')) {
        this.fk_z = initObj.fk_z
      }
      else {
        this.fk_z = 0.0;
      }
      if (initObj.hasOwnProperty('fk_phi')) {
        this.fk_phi = initObj.fk_phi
      }
      else {
        this.fk_phi = 0.0;
      }
      if (initObj.hasOwnProperty('fk_theta')) {
        this.fk_theta = initObj.fk_theta
      }
      else {
        this.fk_theta = 0.0;
      }
      if (initObj.hasOwnProperty('fk_psi')) {
        this.fk_psi = initObj.fk_psi
      }
      else {
        this.fk_psi = 0.0;
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
    // Serializes a message object of type CheckKinFKResponse
    // Serialize message field [state_x]
    bufferOffset = _serializer.float32(obj.state_x, buffer, bufferOffset);
    // Serialize message field [state_y]
    bufferOffset = _serializer.float32(obj.state_y, buffer, bufferOffset);
    // Serialize message field [state_z]
    bufferOffset = _serializer.float32(obj.state_z, buffer, bufferOffset);
    // Serialize message field [state_phi]
    bufferOffset = _serializer.float32(obj.state_phi, buffer, bufferOffset);
    // Serialize message field [state_theta]
    bufferOffset = _serializer.float32(obj.state_theta, buffer, bufferOffset);
    // Serialize message field [state_psi]
    bufferOffset = _serializer.float32(obj.state_psi, buffer, bufferOffset);
    // Serialize message field [fk_x]
    bufferOffset = _serializer.float32(obj.fk_x, buffer, bufferOffset);
    // Serialize message field [fk_y]
    bufferOffset = _serializer.float32(obj.fk_y, buffer, bufferOffset);
    // Serialize message field [fk_z]
    bufferOffset = _serializer.float32(obj.fk_z, buffer, bufferOffset);
    // Serialize message field [fk_phi]
    bufferOffset = _serializer.float32(obj.fk_phi, buffer, bufferOffset);
    // Serialize message field [fk_theta]
    bufferOffset = _serializer.float32(obj.fk_theta, buffer, bufferOffset);
    // Serialize message field [fk_psi]
    bufferOffset = _serializer.float32(obj.fk_psi, buffer, bufferOffset);
    // Serialize message field [correct]
    bufferOffset = _serializer.bool(obj.correct, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CheckKinFKResponse
    let len;
    let data = new CheckKinFKResponse(null);
    // Deserialize message field [state_x]
    data.state_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [state_y]
    data.state_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [state_z]
    data.state_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [state_phi]
    data.state_phi = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [state_theta]
    data.state_theta = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [state_psi]
    data.state_psi = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [fk_x]
    data.fk_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [fk_y]
    data.fk_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [fk_z]
    data.fk_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [fk_phi]
    data.fk_phi = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [fk_theta]
    data.fk_theta = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [fk_psi]
    data.fk_psi = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [correct]
    data.correct = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 49;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/CheckKinFKResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6dd4bd6e62545926c2a7502c0ee7c4f1';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 state_x
    float32 state_y
    float32 state_z
    float32 state_phi
    float32 state_theta
    float32 state_psi
    float32 fk_x
    float32 fk_y
    float32 fk_z
    float32 fk_phi
    float32 fk_theta
    float32 fk_psi
    bool correct
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CheckKinFKResponse(null);
    if (msg.state_x !== undefined) {
      resolved.state_x = msg.state_x;
    }
    else {
      resolved.state_x = 0.0
    }

    if (msg.state_y !== undefined) {
      resolved.state_y = msg.state_y;
    }
    else {
      resolved.state_y = 0.0
    }

    if (msg.state_z !== undefined) {
      resolved.state_z = msg.state_z;
    }
    else {
      resolved.state_z = 0.0
    }

    if (msg.state_phi !== undefined) {
      resolved.state_phi = msg.state_phi;
    }
    else {
      resolved.state_phi = 0.0
    }

    if (msg.state_theta !== undefined) {
      resolved.state_theta = msg.state_theta;
    }
    else {
      resolved.state_theta = 0.0
    }

    if (msg.state_psi !== undefined) {
      resolved.state_psi = msg.state_psi;
    }
    else {
      resolved.state_psi = 0.0
    }

    if (msg.fk_x !== undefined) {
      resolved.fk_x = msg.fk_x;
    }
    else {
      resolved.fk_x = 0.0
    }

    if (msg.fk_y !== undefined) {
      resolved.fk_y = msg.fk_y;
    }
    else {
      resolved.fk_y = 0.0
    }

    if (msg.fk_z !== undefined) {
      resolved.fk_z = msg.fk_z;
    }
    else {
      resolved.fk_z = 0.0
    }

    if (msg.fk_phi !== undefined) {
      resolved.fk_phi = msg.fk_phi;
    }
    else {
      resolved.fk_phi = 0.0
    }

    if (msg.fk_theta !== undefined) {
      resolved.fk_theta = msg.fk_theta;
    }
    else {
      resolved.fk_theta = 0.0
    }

    if (msg.fk_psi !== undefined) {
      resolved.fk_psi = msg.fk_psi;
    }
    else {
      resolved.fk_psi = 0.0
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
  Request: CheckKinFKRequest,
  Response: CheckKinFKResponse,
  md5sum() { return '6dd4bd6e62545926c2a7502c0ee7c4f1'; },
  datatype() { return 'scara_command/CheckKinFK'; }
};
